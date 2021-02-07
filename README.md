# DRL-SFCP

This is the implement of "DRL-SFCP: Adaptive Service Function Chains Placement with Deep Reinforcement Learning" which is accepted by ICC 2021.

## Forder Structure

```Plain
drl-sfcp
├─data
│  ├─network.py
│  ├─physical_network.py
│  ├─service_fuction_chain.py
│  ├─sfc_simulator.py
│  └─utils.py
│
├─algo
│  ├─environment.py
│  ├─drl_sfcp_env.py
│  ├─agent.py
│  └─test.py
│
├─main.py
```

## `data` Module

- `Network`: A class Inherited from networkx.Graph so as to sufficient reusability and flexible customization.
- `PhysicalNetwork`: A class implements the physical network.
- `ServiceFuctionChain`: A class implements the service fuction chain.
- `sfc_simulator`: A class implements the sfc simulator which generates SFCs following the configuration of conditions.

## `algo` Module

Under reconstruction and will come here as soon as possible...
