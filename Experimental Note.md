# *Experimental* Note of My Paper

## 1. Standard Conditions of Test Dataset

In this section, we analyze the results of  experiment in different conditions using **GRC** algorithm, whose purpose is looking for the standard test dataset which is suitable for our experiment.

### 1.1 Physical Network

|                                      | Setting       |
| ------------------------------------ | ------------- |
| Nodes Number                         | 100           |
| Edges Number                         | about 500     |
| Nodes Attributions                   | CPU, RAM, ROM |
| Edges Attributions                   | Bandwidth     |
| Total Resource of Every Attributions | [50, 100]     |

### 1.2 Virtual Network Dataset

The standard of dataset we desire:

- In the initial settings, the GRC don's cope with the dataset very well, which contribute to highlighting the superiority of our algorithm. (The acceptance ratio is expected to be distributed in [60%, 70%]).

- More importantly, with one factor making a more minor adjustment , the experiment result (i.e. acceptance ratio) can't have a tremendous change, which is unbeneficial to conduct more comparative experiments in a series of changing conditions.

- Only taking the impact of arrival rate, we hope to conduct 8 experiments in two constraints: (a) From an average of 10 arrivals per 100 time units to 24 arrivals with an increasing step size by 2. (b) Within the range of arrival rate change, the acceptance radio is in the range of [50%, 100].

Test on GRC for Looking Suitable Dataset

#### 1.2.1. Impact of  GRC_d​

| GRC_d | SFC Number | VNF Number| Resource Request | Average Lifetime| **Arrival Rate** | **Acceptance Ratio** | **Average Revenue** | Revenue-to-Cost |
| - | - | - | - | - | - | - | - | - |
| 0.1 | 2000 | [2 ~ 15] | [2 ~ 30] | 500 | 12 | 0.749 | 343.582 | 0.7617955904003406 |
| 0.5 | 2000 | [2 ~ 15] | [2 ~ 30] | 500 | 12 | 0.7825 | 358.2555 | 0.7706930322459898 |
| 0.9 | 2000 | [2 ~ 15] | [2 ~ 30] | 500 | 12 | 0.8075 | 371.025 | 0.7821673641049425 |

The data of the table shows the effect of $d$ on GRC performance. With the increase of $d$ value, the performance of GRC is improved.

#### 1.2.2. Impact of  Arrival Rate and Average Lifetime

| GRC_d | SFC Number | VNF Number| Resource Request | Average Lifetime| Arrival Rate | **Acceptance Ratio** | **Average Revenue** | Revenue-to-Cost |
| - | - | - | - | - | - | - | - | - |
| 0.1 | 2000 | [2 ~ 15] | [2 ~ 30] | 500 | 10 | 0.846 | 401.361 | 0.763 |
| 0.1 | 2000 | [2 ~ 15] | [2 ~ 30] | 500 | 12 | 0.749 | 343.582 | 0.762 |
| 0.1 | 2000 | [2 ~ 15] | [2 ~ 30] | 500 | 14 | 0.667 | 303.468 | 0.760 |
| 0.1 | 2000 | [2 ~ 15] | [2 ~ 30] | 400 | 10 | 0.952 | 466.786 | 0.767 |
| 0.1 | 2000 | [2 ~ 15] | [2 ~ 30] | 400 | 12 | 0.875 | 422.675 | 0.762 |
| 0.1 | 2000 | [2 ~ 15] | [2 ~ 30] | 400 | 14 | 0.821 | 388.595 | 0.762 |
| 0.1 | **2000** | **[2 ~ 15]** | **[2 ~ 30]** | **400** | **20** | **0.625** | **283.147** | **0.756** |
| 0.1 | 2000 | [2 ~ 15] | [2 ~ 30] | 400 | 24 | 0.546 | 236.001             | 0.753 |

As the table shows, when the arrival rate changes, with the average lifetime increasing, the variations in acceptance ratio are fiercer. Under the condition of  Average Lifetime = 400​, Arrival Rate in [10, 24] => Acceptance Ratio in [0.55, 0.95] using GRC.

#### 1.2.3 Basic Experimental Settings of Virtual Network

Finally, the basic experimental conditions as follow:

| Condition        | Setting  |
| ---------------- | -------- |
| SFC Number       | 2000     |
| VNF Number       | [2 ~ 15] |
| Resource Request | [2 ~ 30] |
| Average Lifetime | 400      |
| **Arrival Rate** | **20**   |

Arrival Rate is a variable in the range of [10, 24]

#### 1.2.4 Preview Test

| **Arrival Rate** | Algorithm              | Acceptance Ratio | Average Revenue | Revenue-to-Cost |
| ---------------- | ---------------------- | ---------------- | --------------- | --------------- |
| 20               | SFC-GRC                | 0.626            | 283.316         | 0.756           |
| 20               | Random                 | 0.648            | 283.6755        | 0.758           |
| 20               | The Undertrained Model | 0.7305           | 333.483         | 0.829           |

Compared the SFC-GRC and Random, our model trained  by old and simple dataset (about 5000 SFC in not crowed condition) shown a slight improvement, which proves the effectiveness of our model preliminary.

## 2. Model Performance Verification

In this section, we test our model performance by modify the architecture mildly.

### 2.1. Basic Parameters Value

| Parameters | Value |
| ---------- | ----- |
| batch size | 16    |
| epoch      | 2     |

After using the same settings to train models for 2 epochs, the performance of different architecture  models will be verified testing on the same dataset.

### 2.2. The State Input of Critic

- **Only PN**: Only considering the state of physical network as the input of critic network.
- **PN + VN**: Input the state of both physical network and  virtual network to critic network.

| State Setting | Dataset | Acceptance Ratio | Average Revenue | Revenue-to-Cost |
| ------------- | ------- | ---------------- | --------------- | --------------- |
| Only PN       | train   |                  |                 |                 |
| PN + VN       | train   |                  |                 |                 |
| Only PN       | test    |                  |                 |                 |
| PN + VN       | test    |                  |                 |                 |

### 2.3. The Impact of GCN Layers Number

| GCN Layers Number | Dataset | Acceptance Ratio | Average Revenue | Revenue-to-Cost |
| ------------- | ---------- | --------------- | --------------- | ------------- |
| 1          | train     |                  |                 |                 |
| 2       | train  |                  |                 |                 |
| 1 | test | | | |
| 2 | test | | | |

### 2.4. The Impact of Dropout Layer

| Dropout Layer | Dataset | Acceptance Ratio | Average Revenue | Revenue-to-Cost |
| ------------- | ------- | ---------------- | --------------- | --------------- |
| With          | train   |                  |                 |                 |
| Without       | train   |                  |                 |                 |
| With          | test    |                  |                 |                 |
| Without       | test    |                  |                 |                 |

### 2.5. The Impact of TD Gamma

| TD Gamma | Dataset | Acceptance Ratio | Average Revenue | Revenue-to-Cost |
| -------- | ------- | ---------------- | --------------- | --------------- |
| 0        | train   |                  |                 |                 |
| 0.95     | train   |                  |                 |                 |
| 0        | test    |                  |                 |                 |
| 0.95     | test    |                  |                 |                 |

8  0.72
12 0.70

### 2.6. The Impact of Learning Rate

| Learning Rate (actor/critic) | Dataset | Acceptance Ratio | Average Revenue | Revenue-to-Cost |
| ---------------------------- | ------- | ---------------- | --------------- | --------------- |
| 0.00025 \| 0.0005            | train   |                  |                 |                 |
| 0.00025 \| 0.0005            | train   |                  |                 |                 |
| 0.000025 \| 0.00005          | test    |                  |                 |                 |
| 0.000025 \| 0.00005          | test    |                  |                 |                 |

## 3. Train Our Model

## 4. Comparative Algorithm

- SFC-GRC:

- SFC-MCTS:

## 5. Experiment Result

### Acceptance Radio and Average Revenue

| **Arrival Rate** | Algorithm    | Acceptance Ratio | Average Revenue | Revenue-to-Cost |
| ---------------- | ------------ | ---------------- | --------------- | --------------- |
| 10               | SFC-GRC      | 0.952            | 466.786         | 0.767           |
| *10*             | *SFC-MCTS*   |                  |                 |                 |
| **10**           | **DRL-SFCP** |                  |                 |                 |
| 12               | SFC-GRC      | 0.875            | 422.675         | 0.762           |
| *12*             | *SFC-MCTS*   |                  |                 |                 |
| **12**           | **DRL-SFCP** |                  |                 |                 |
| 14               | SFC-GRC      | 0.821            | 388.595         | 0.762           |
| *14*             | *SFC-MCTS*   |                  |                 |                 |
| **14**           | **DRL-SFCP** |                  |                 |                 |
| 16               | SFC-GRC      | 0.75             | 351.408         | 0.760           |
| *16*             | *SFC-MCTS​*   |                  |                 |                 |
| **16**           | **DRL-SFCP** |                  |                 |                 |
| 18               | SFC-GRC      | 0.696            | 319.256         | 0.759           |
| *18*             | *SFC-MCTS*   |                  |                 |                 |
| **18**           | **DRL-SFCP** |                  |                 |                 |
| 20               | SFC-GRC      | 0.626            | 283.316         | 0.756           |
| *20*             | *SFC-MCTS*   |                  |                 |                 |
| **20**           | **DRL-SFCP** |                  |                 |                 |
| 22               | SFC-GRC      | 0.589            | 261.630         | 0.755           |
| *22*             | *SFC-MCTS*   |                  |                 |                 |
| **22**           | **DRL-SFCP** |                  |                 |                 |
| 24               | SFC-GRC      | 0.546            | 236.001         | 0.753           |
| *24*             | *SFC-MCTS*   |                  |                 |                 |
| **24**           | **DRL-SFCP** |                  |                 |                 |
|                  |              |                  |                 |                 |
