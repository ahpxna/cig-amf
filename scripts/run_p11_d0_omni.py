"""Paired OmniArena D0 existence/heterogeneity screen.

The adaptive policy below is prespecified and deliberately is not CIG-AMF.
Thus even a confirmatory PASS only authorizes subsequent theorem/algorithm
work; it cannot support a learned-policy or causal-identification claim.
"""
from __future__ import annotations
import argparse, copy, hashlib, json, pickle
from dataclasses import asdict
from pathlib import Path
import numpy as np
from envs.omni_arena import OmniArena
from models.disturbance_contracts import DisturbanceRegime, PairedDisturbanceRecord, adjudicate_d0
from scripts.scientific_gate_common import atomic_json

PROTOCOL_VERSION = "p11_d0_omni_paired_regime_v1"
ARMS = {"force_left": OmniArena.LEFT, "force_right": OmniArena.RIGHT}
METRICS = ("future_state_distance", "future_response_shift", "future_policy_distance")

def _sha(value): return hashlib.sha256(pickle.dumps(value, protocol=4)).hexdigest()
def _softmax(row):
    row = np.asarray(row, dtype=np.float64); values = np.exp(row - np.max(row)); return values / values.sum()

class OnlinePolicy:
    """Fixed D0 adaptive surrogate; no CIG-AMF parameters are consumed."""
    def __init__(self, env, learning_rate=0.2, adaptive_weight=0.35):
        self.preferences = np.zeros((env.n_agents, env.N_ACTIONS), dtype=np.float64)
        self.learning_rate, self.adaptive_weight = float(learning_rate), float(adaptive_weight)
    def clone(self):
        other = object.__new__(OnlinePolicy); other.preferences = self.preferences.copy()
        other.learning_rate, other.adaptive_weight = self.learning_rate, self.adaptive_weight; return other
    def actions(self, env, live):
        chosen = []
        for agent in range(env.n_agents):
            scripted = np.asarray(env.scripted_policy_distribution(agent), dtype=np.float64)
            probs = scripted if not live else ((1-self.adaptive_weight)*scripted + self.adaptive_weight*_softmax(self.preferences[agent]))
            chosen.append(int(np.argmax(probs)))
        return chosen
    def update(self, actions, rewards):
        for agent, (action, reward) in enumerate(zip(actions, rewards)):
            gradient = -_softmax(self.preferences[agent]); gradient[int(action)] += 1.0
            self.preferences[agent] += self.learning_rate*float(np.clip(reward,-1,1))*gradient
    def distance(self, other):
        first=np.stack([_softmax(x) for x in self.preferences]); second=np.stack([_softmax(x) for x in other.preferences])
        return float(np.mean(0.5*np.abs(first-second).sum(axis=1)))

def _state_vector(snapshot, env):
    values=[]
    for agent in range(env.n_agents): values.extend(np.asarray(snapshot["positions"][agent])/env.grid_size)
    for field in ("gate_open","resource_available","carrying","low_priority_active"):
        values.extend(float(bool(snapshot[field][zone])) for zone in range(env.n_zones))
    values.extend(float(snapshot["active_lane"][zone]=="B") for zone in range(env.n_zones))
    return np.asarray(values,dtype=np.float64)
def _state_distance(first,second,env):
    delta=_state_vector(first,env)-_state_vector(second,env); return float(np.sqrt(np.mean(delta*delta)))

def _branch(env,snapshot,regime,source,outcome,forced_action,horizon):
    env.restore_state(copy.deepcopy(snapshot)); policy=OnlinePolicy(env); initial=policy.clone(); live=regime==DisturbanceRegime.LIVE_LEARNING
    actions=policy.actions(env,live); baseline_action=int(actions[source])
    if forced_action is not None:
        if int(forced_action)==baseline_action: raise RuntimeError("forced action equals baseline action")
        actions[source]=int(forced_action)
    requested=int(actions[source]); _,rewards,done,_=env.step(actions); executed=int(env.last_actions[source])
    if executed!=requested: raise RuntimeError("requested source action was not executed")
    if live: policy.update(actions,rewards)
    if regime==DisturbanceRegime.RESET:
        env.restore_state(copy.deepcopy(snapshot)); policy=initial.clone(); done=False
    future_rewards,future_states=[],[]
    for _ in range(int(horizon)):
        if done: raise RuntimeError("branch terminated before complete horizon")
        actions=policy.actions(env,live); _,rewards,done,_=env.step(actions)
        future_rewards.append(float(rewards[outcome])); future_states.append(copy.deepcopy(env.clone_state()))
        if live: policy.update(actions,rewards)
    return {"baseline_action":baseline_action,"executed_action":executed,"rewards":future_rewards,"states":future_states,"policy":policy}

def collect(*,mode,seeds,replicates_per_seed,horizon,burn_in):
    if mode not in {"quick","confirmatory"}: raise ValueError("mode must be quick or confirmatory")
    if not seeds or len(set(seeds))!=len(seeds): raise ValueError("seeds must be non-empty and unique")
    if mode=="confirmatory" and (len(seeds)<3 or replicates_per_seed<8 or horizon<8):
        raise ValueError("confirmatory requires >=3 seeds, >=8 replicates/seed, horizon>=8")
    target_key="omni/zone0/blocker_to_collector/v1"; records=[]; audit_rows=[]; replicate=0
    for seed in seeds:
        env=OmniArena(n_agents=24,n_zones=4,max_steps=max(40,burn_in+horizon+8),phase_length=1000,causal_horizon=horizon,seed=int(seed),mode="cooperative",structural_factor=False,behavioral_factor=False)
        source=int(env.zone_role_agents[0][env.ROLE_BLOCKER]); outcome=int(env.zone_role_agents[0][env.ROLE_COLLECTOR])
        bank=env.sample_state_bank(n_states=max(8,4*replicates_per_seed),burn_in=burn_in,bank_seed=int(seed)+41177,min_remaining_steps=horizon+1)
        accepted=[]
        for snapshot in bank:
            env.restore_state(copy.deepcopy(snapshot))
            if int(env.scripted_policy(source)) not in set(ARMS.values()): accepted.append(snapshot)
            if len(accepted)==replicates_per_seed: break
        if len(accepted)!=replicates_per_seed: raise RuntimeError("insufficient states with two nonbaseline fixed arms")
        for local_id,snapshot in enumerate(accepted):
            for regime in DisturbanceRegime:
                baseline=_branch(env,snapshot,regime,source,outcome,None,horizon)
                for arm,action in ARMS.items():
                    intervention=_branch(env,snapshot,regime,source,outcome,action,horizon)
                    row=PairedDisturbanceRecord(regime=regime,arm=arm,replicate=replicate,target_key=target_key,immediate_cost=1.0,
                        future_state_distance=float(np.mean([_state_distance(a,b,env) for a,b in zip(baseline["states"],intervention["states"])])),
                        future_response_shift=float(abs(np.mean(intervention["rewards"])-np.mean(baseline["rewards"]))),
                        future_policy_distance=baseline["policy"].distance(intervention["policy"]) if regime==DisturbanceRegime.LIVE_LEARNING else 0.0)
                    records.append(row); audit_rows.append({**asdict(row),"regime":regime.value,"seed":int(seed),"local_replicate":local_id,
                        "start_state_sha256":_sha(snapshot),"source_agent":source,"outcome_agent":outcome,"baseline_source_action":baseline["baseline_action"],
                        "requested_forced_action":int(action),"executed_forced_action":intervention["executed_action"],"action_verified":intervention["executed_action"]==int(action),"paired_future_steps":horizon})
            replicate+=1
    result=adjudicate_d0(records,metric_thresholds=dict.fromkeys(METRICS,1e-5),minimum_arm_spreads=dict.fromkeys(METRICS,1e-6),immediate_cost_tolerance=0.0,minimum_replicates=len(seeds)*replicates_per_seed)
    return {"protocol_version":PROTOCOL_VERSION,"mode":mode,"collection_complete":True,"d0_status":result.status if mode=="confirmatory" else "SMOKE_ONLY",
        "raw_d0_adjudication":result.status,"evidence_class":"PRESPECIFIED_D0_EXISTENCE_SCREEN_ONLY" if mode=="confirmatory" else "PLUMBING_SMOKE_ONLY",
        "reasons":list(result.reasons),"records":audit_rows,"regime_metric_means":result.regime_metric_means,"arm_metric_means":result.arm_metric_means,
        "manifest":{"fixed_target_key":target_key,"fixed_arm_actions":ARMS,"record_count":len(records),"immediate_cost_definition":"one verified source-action override",
        "immediate_cost_matching":"EXACT_BY_DESIGN","crn_protocol":"identical complete clone including NumPy RNG state","metric_aggregation":"SEPARATE_ONLY_NO_MIXED_UNIT_SCALAR",
        "live_policy_scope":"prespecified D0 surrogate, not CIG-AMF","claim_limit":"D0 only; no correctness or sample-complexity theorem"}}

def main(argv=None):
    parser=argparse.ArgumentParser(); modes=parser.add_mutually_exclusive_group(required=True); modes.add_argument("--quick",action="store_true"); modes.add_argument("--confirmatory",action="store_true")
    parser.add_argument("--seeds",type=int,nargs="+",default=[0,1,2]); parser.add_argument("--replicates-per-seed",type=int); parser.add_argument("--horizon",type=int); parser.add_argument("--burn-in",type=int,default=2); parser.add_argument("--out",required=True); args=parser.parse_args(argv)
    mode="quick" if args.quick else "confirmatory"; seeds=tuple(args.seeds[:1] if mode=="quick" else args.seeds); replicates=args.replicates_per_seed or (2 if mode=="quick" else 8); horizon=args.horizon or (4 if mode=="quick" else 8)
    try: payload=collect(mode=mode,seeds=seeds,replicates_per_seed=replicates,horizon=horizon,burn_in=args.burn_in); code=0
    except Exception as error:
        payload={"protocol_version":PROTOCOL_VERSION,"mode":mode,"collection_complete":False,"d0_status":"FAILED","raw_d0_adjudication":"FAILED","evidence_class":"NONE","reasons":[f"collector failed closed: {error}"],"records":[]}; code=2
    atomic_json(Path(args.out),payload); print(json.dumps({"out":str(Path(args.out).resolve()),"collection_complete":payload["collection_complete"],"d0_status":payload["d0_status"],"records":len(payload["records"])},indent=2)); return code
if __name__=="__main__": raise SystemExit(main())
