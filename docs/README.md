take note: 
* parameter trip r belongs to a vehicle. for a node, vehicle Va1 might make 3 trips to it whereas Va2 only makes 1
* created a waiting time variable unique to each vehicle/trip, limited it to [0, 30] because without this, there will be many solutions

current problems:
* dell proj laptop cpu processing speed too slow
* unable to determine value of max_trips at the start, can give an estimate, but too high a value increases computational time exponentially
* cannot include max/min function in objective function, so right now is total arrival time
* unable to constraint launch-off nodes of alpha/beta vehicles from destination/start points, but not a concern if we assume the vehicles are ready at their respective locations
<br>



# Introduction
## Overview
This project aims to optimise a two-part transportation system, as shown below: 

![overview](images/overview.jpg "Overview"){width=50%}

*Note that the image above is for illustration purposes only, and the [waypoints can be variables](#location-constraints).*

The system should be optimised such that the following objectives are achieved:
1. The total time taken to transport all personnel from START to DESTINATION is minimised 
2. The average wait time of each personnel in between the two modes of transportation (i.e at HOLDING) is minimised.

## Approach
Intuitively, one would break down the problem into two overlapping scheduling problems - one for Channel Beta, and the other for Channel Alpha. This approach can be feasible if we are only looking to optimise the total time taken. In that case, by optimising the time taken by Channel Beta and Alpha respectively, we will inadvertently find the best solution to the overall problem.

However, this is not the only metric we are looking to optimise. In order to minimise the wait time as well, Channel Beta might require a suboptimal solution. For example, if Channel Beta can transport all personnel in 2 hours and Channel Alpha can transport all personnel in 10 hours, then the average wait time will be 8 hours if Channel Beta were to transport all personnel as quickly as possible. Rather, one can deduce that the optimal solution for Channel Beta in this case will be to feed personnel gradually into the HOLDING at a rate that matches the processing rate of Channel Alpha. However, how could we formulate an objective function that encapsulates this? The answer is we probably cannot, unless we view this scheduling problem end-to-end.

With that being said, this scheduling problem does contain a sub-problem - route optimization. Since Channel Beta is on land, the travelling time required can vary significantly depending on the route chosen. As such, route optimisation can help to reduce the total amount of time taken by personnel to travel from START to HOLDING, which indirectly reduces the amount of time taken in total by personnel. 

# Problem Statement
We break the problems into two sub-problems. 
* Schedule Optimisation - Determine the transport time and waiting time of every personnel
* Route Optimisation - Determine the optimal route between any two locations

## Scheduling Optimisation
We assume that Route Optimisation has been completed, and we have with us a list of routes between each location and the travelling time required. The list of possible routes is denoted by $A$.

### <b> 1. Decision variables </b>
Looking at the two metrics to be minimised, we can see that both requires scheduling information of all personnel. Therefore, for each of the $N$ personnel, the details of their trips are shown in the following table: 

| Personnel $n$ | START $P_S$ | HOLDING $P_H$ | DESTINATION $P_D$ | Beta Vehicle $\beta$ | Alpha Vehicle $\alpha$ | START Departure $T_1$ (mins) | HOLDING Arrival $T_2$ (mins) | HOLDING Departure $T_3$ (mins) | DESTINATION Arrival $T_4$ (mins) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | A | 1 | X | 01 | 01 | 0 | 60 | 80 | 100 |
| 2 | A | 1 | X | 01 | 01 | 0 | 60 | 80 | 100 |
| ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |
| N | B | 2 | Z | 05 | 02 | 120 | 170 | 210 | 230 |

This schedule shows the arrival and departure time (in minutes) after an arbitrary time, taken as 0. Taking personnel $N$ as example, if the schedule starts at 0700, he will depart START point B at 0900 to reach HOLDING point 2 at 0950.

<br>
<br>
<br>
<br>

$$
\begin{align*}
x_{ij}^{nvr} &= 
\begin{cases}
1 & \text{if personnel $n$ takes trip r of alpha vehicle v through i, j}\\
0 & \text{otherwise}
\end{cases} \\

y_{ij}^{nvr} &= 
\begin{cases}
1 & \text{if personnel $n$ takes trip r of beta vehicle v through i, j}\\
0 & \text{otherwise}
\end{cases} \\

z_i^n &= 
\begin{cases}
1 & \text{if personnel $n$ visits node $i$}\\
0 & \text{otherwise}
\end{cases} \\

p_{ij}^{vr} &= 
\begin{cases}
1 & \text{if trip r of alpha vehicle v passes through $i, j$}\\
0 & \text{otherwise}
\end{cases} \\

q_{ij}^{vr} &= 
\begin{cases}
1 & \text{if trip r of beta vehicle v passes through $i, j$}\\
0 & \text{otherwise}
\end{cases} \\

w_i^{vr} &= 
\begin{cases}
1 & \text{if trip r of alpha vehicle v visits node $i$}\\
0 & \text{otherwise}
\end{cases} \\

T_i^{vr} &= \text{arrival time of trip r of vehicle v at node $i$} \\

t_i^{n} &= \text{arrival time of personnel n at node $i$} \\

b_i^{vr} \in \Z^+  &= \text{wait time of trip r of vehicle v at node $i$}, b_i^{vr} \\

\end{align*}
$$

<b> Notations </b> <br>
$$
\begin{align*}
R & = \text{set of trips} \\
V_A & = \text{Alpha vehicles} \\
V_B & = \text{Beta vehicles} \\
P & = P_S \cup P_H \cup P_D \\
A & = \text{all combinations of routes i, j} \\
d_{ij} & = \text{travel time of route i, j} \\
l & = \text{loading and unloding time} \\
Q_v &= \text{capacity constraints for vehicle v} \\
\end{align*}
$$

<b> Total time </b> <br>
We can subtract the earliest START Departure time from the latest DESTINATION Arrival time:
$$
f_1(x) = \max_i a_i - \min_j d_j \quad i \in P_D, j \in P_S
$$

<b> Average wait time </b> <br>
We take the mean difference in time between HOLDING Departure time and HOLDING Arrival time:
$$
f_2(x) = \dfrac{1}{N} \sum_{i \in P_H} \sum_{j \in P_D} \sum_{k \in P_S} \sum_{n \in N} (x_{ij}^n d_i - y_{ki}^n a_i)
$$

### <b> 2. Objective Function </b>
We aim to minimise a combination of both functions above, with $\lambda$ being a deterministic weight assigned to the second function. 
$$
f(x) = f_1(x) + \lambda f_2(x) \qquad \lambda \in (0, 1)
$$

### <b> 3. Parameters </b>
Below are the fixed values determined prior to running the optimisation algorithm.

| Parameter | Description | Remarks |
| --- | --- | --- |
| N | Number of Personnel |  |
| R | Maximum number of trips per vehicle |  |
| $P_S$ | Number of Start points |  |
| $P_H$ | Number of Holding points |  |
| $P_D$ | Number of Destination points |  |
| $V_A$ | Number of alpha vehicles | for transporting personnel or equipment |
| $V_B$ | Number of beta vehicles | non-transportation assets |
| $Q_V$ | Load Capacity for each vehicle |  |
| $l$ | Loading and Unloading time |  |
| $t_{ij}^v$ | Travelling time of vehicle v from node i to j | json format |


### <b> 4. Constraints </b> 
The following constraints keep the values of the decision variables in check.

<b> i. Vehicle v arrives and departs from same holding node if node is visited during that trip </b> <br>
$$
\begin{align*}
\sum_{j \in P_D} p_{ji}^{vr} = \sum_{j \in P_D} p_{ij}^{vr} = w_i^{vr} & \quad \forall v \in V_A, r \in R, i \in P_H \\
\sum_{j \in P_S} q_{ji}^{vr} = \sum_{j \in P_S} q_{ij}^{vr} = w_i^{vr} & \quad \forall v \in V_B, r \in R, i \in P_H \\
\end{align*}
$$

<b> ii. Vehicle v departs from node i (start/destination) in trip r+1, if trip exsits, if it arrives at node i in trip r </b> <br>
$$
\begin{align*}
w_i^{vr} = \sum_{j \in P_H} p_{ji}^{vr} \geq \sum_{j \in P_H} p_{ij}^{vr+1} = w_i^{vr+1} & \quad \forall v \in V_A, r \in R, i \in P_D \\
w_i^{vr} = \sum_{j \in P_H} p_{ji}^{vr} \geq \sum_{j \in P_H} p_{ij}^{vr+1} = w_i^{vr+1} & \quad \forall v \in V_B, r \in R, i \in P_S \\
\end{align*}
$$

<b> iii. Only 1 start<->holding / holding<->destination route per trip per vehicle </b> <br>
$$
\begin{align*}
\sum_{j \in P_H} p_{ij}^{vr} <= 1 & \quad \forall v \in V_A, r \in R, i \in P_D \\
\sum_{j \in P_D} p_{ij}^{vr} <= 1 & \quad \forall v \in V_A, r \in R, i \in P_H \\
\sum_{j \in P_S} q_{ij}^{vr} <= 1 & \quad \forall v \in V_B, r \in R, i \in P_H \\
\sum_{j \in P_H} q_{ij}^{vr} <= 1 & \quad \forall v \in V_B, r \in R, i \in P_S \\
\end{align*}
$$

<b> iv. Personnel arrives and departs from same holding node if node is visited </b> <br>
$$
\sum_{v \in V_B} \sum_{r \in R} \sum_{j \in P_S} y_{ji}^{nvr} = \sum_{v \in V_A} \sum_{r \in R} \sum_{j \in P_D} x_{ij}^{nvr} = z_i^n \quad \forall n \in N, i \in P_H \\
$$

<b> v. Personnel visits holding/destination node only if it boards the vehicle </b> <br>
$$
\begin{align*}
\sum_{v \in V_A} \sum_{r \in R} \sum_{j \in P_S} x_{ji}^{nvr} p_{ji}^{vr} = z_i^n & \quad \forall n \in N, i \in P_H \\
\sum_{v \in V_B} \sum_{r \in R} \sum_{j \in P_H} y_{ji}^{nvr} q_{ji}^{vr} = z_i^n & \quad \forall n \in N, i \in P_D \\
\end{align*}
$$

<b> vi. Personnel departs from start node if it is stationed there </b> <br>
$$
\begin{align*}
\sum_{v \in V_B} \sum_{r \in R} \sum_{j \in P_H} y_{ij}^{nvr} q_{ij}^{vr} = z_i^n & \quad \forall n \in N, i \in P_S \\
\end{align*}
$$

<b> vii. Only 1 start->holding & holding->destination per personnel </b> <br>
$$
\begin{align*}
\sum_{v \in V_A} \sum_{r \in R} \sum_{i \in P_H} \sum_{j \in P_D} x_{ij}^{nvr} = 1 & \quad \forall n \in N \\
\sum_{v \in V_B} \sum_{r \in R} \sum_{i \in P_S} \sum_{j \in P_H} y_{ij}^{nvr} = 1 & \quad \forall n \in N \\
\end{align*}
$$


<b> viii. Vehicle capacity constraints </b> <br>
$$
\begin{align*}
\sum_{n \in N} x_{ij}^{nvr} p_{ij}^{vr} \leq Q_v & \quad \forall v \in V_A, r \in R, i \in P_H, j \in P_D \\
\sum_{n \in N} y_{ij}^{nvr} q_{ij}^{vr} \leq Q_v & \quad \forall v \in V_B, r \in R, i \in P_S, j \in P_H
\end{align*}
$$

<b> ix. Arrival time at Start/Destination >= Arrival time at Holding + Loading time + Waiting time + Travel time if vehicle v takes route during trip r </b> <br>
$$
\begin{align*}
T_i^{vr} \geq \sum_{j \in P_H} p_{ji}^{vr} (T_j^{vr} + l + b_j^{vr} + d_{ji}) & \quad \forall v \in V, r \in R, i \in P_D \\
T_i^{vr} \geq \sum_{j \in P_H} q_{ji}^{vr} (T_j^{vr} + l + b_j^{vr} + d_{ji}) & \quad \forall v \in V, r \in R, i \in P_S \\
\end{align*}
$$
*$T_j^{vr}$ and $b_j^{vr}$ is a different variable for alpha/beta vehicles

<b> x. Arrival time of trip r+1 at Holding >= Arrival time at Start/Destination of trip r + Loading time + Waiting time + Travel time if vehicle v takes route </b> <br>
$$
\begin{align*}
T_i^{vr+1} = \sum_{j \in P_D} p_{ji}^{vr+1} (T_j^{vr} + l + b_j^{vr} + d_{ji}) & \quad \forall v \in V_A, r \in R, i \in P_H \\
T_i^{vr+1} = \sum_{j \in P_S} q_{ji}^{vr+1} (T_j^{vr} + l + b_j^{vr} + d_{ji}) & \quad \forall v \in V_B, r \in R, i \in P_H \\
\end{align*}
$$

<b> xi. Arrival time of trip 0 at Holding >= Loading time + Waiting time + Travel time if vehicle v takes route </b> <br>
$$
\begin{align*}
T_i^{v0} = \sum_{j \in P_D} p_{ji}^{v0} (l + b_j^{vr} + d_{ji}) & \quad \forall v \in V_A, i \in P_H \\
T_i^{v0} = \sum_{j \in P_S} q_{ji}^{v0} (l + b_j^{vr} + d_{ji}) & \quad \forall v \in V_B, i \in P_H \\
\end{align*}
$$

<b> xii. Arrival time of personnel n at node i = Arrival time of trip r of vehicle v at node i if personnel boards it </b> <br>
$$
\begin{align*}
\sum_{v \in V_A} \sum_{r \in R} x_{ji}^{nvr} T_i^{vr} = t_i^n & \quad \forall n \in N, i \in P_D \\
\sum_{v \in V_B} \sum_{r \in R} y_{ji}^{nvr} T_i^{vr} = t_i^n & \quad \forall n \in N, i \in P_S \\
\end{align*}
$$
*don't need to multiply by p or q because each trip of each vehicle can only have 1 arrival time at any node

<b> xiii. Personnel can only board trip r of alpha vehicle if arrival time of trip r of alpha vehicle at Destination node - time taken for route > arrival time of personnel n at holding node i </b> <br>
$$
\begin{align*}
\sum_{v \in V_A} \sum_{r \in R} \sum_{j \in P_D} x_{ij}^{nvr} (T_j^{vr} - d_{ij}) \geq t_i^n & \quad \forall n \in N, i \in P_H \\
\end{align*}
$$

<b> Integer constraints </b> <br>
$$
\begin{align*}
x_{ij}^n, y_{ij}^n, z_i^n \in \set{0, 1} & \quad \forall n \in N, i \in P, (i,j) \in A \\
p_{ij}, q_{ij}, w_i \in \set{0, 1} & \quad \forall i \in P, (i,j) \in A \\
T_i^{vr}, t_i^n \in Z^+ & \quad \forall i \in P
\end{align*}
$$

<b> Constraints notes </b> <br>
* don't need constraint whereby personnel can't travel backwards because no such variable with e.g. P_H to P_S
* don't need constraint wherby personnel boards vehicle only if it is at that node because we have constraint of personnel leaving and exiting same holding node if visited

#### <b> (Unconfirmed) </b>

<b> Subsequent trips of vehicle r can only take place if prior trip of the same route is taken </b> <br>
$$
\begin{align*}
p_{ij}^{vr+1} >= p_{ij}^{vr} & \quad \forall v \in V_A, r \in R, (i,j) \in P_H \cup P_D \\
q_{ij}^{vr+1} >= q_{ij}^{vr} & \quad \forall v \in V_B, r \in R, (i,j) \in P_S \cup P_H \\
\end{align*}
$$

<b> Vehicle v arrives at node i for the same number of trips that depart from node i (start/destination) </b> <br>
$$
\begin{align*}
\sum_{r \in R} \sum_{j \in P_H} p_{ji}^{vr} = \sum_{r \in R} \sum_{j \in P_H} p_{ij}^{vr} = \sum_{r \in R} w_i^{vr} & \quad \forall v \in V_A, i \in P_D \\
\sum_{r \in R} \sum_{j \in P_H} q_{ji}^{vr} = \sum_{r \in R} \sum_{j \in P_H} q_{ij}^{vr} = \sum_{r \in R} w_i^{vr} & \quad \forall v \in V_B, i \in P_S \\
\end{align*}
$$

## Route Optimisation
In this case, we are simply looking for the shortest-path (in terms of time, not distance) between two points, which can be fully solved using Djikstra's Algorithm. 

### Problem Statement
The problem can be formalised as such - given a graph $G$, what is the shortest path from point $a$ to $b$? For our use case, we can represent the map as a graph, where each node is a traffic junction and each edge is a road. The weight of each edge is the time taken to travel along the road.

### Solution
Applying Djikstra's algorithm, we can find the optimal path (and also the optimal time taken):
$$
D(G, a, b) \rightarrow P \qquad P \subset N
$$

To make the system more dynamic, $G$ can be updated in realtime based on the road and traffic conditions. 

# Proposed Solutions
This problem is a form of Mixed-Integer Nonlinear problem, where $P_S$, $P_D$, $P_H$, $\alpha$ and $\beta$ are discrete values represented by integers, and $T_1$, $T_2$, $T_3$ and $T_4$ are real numbers (even though they can be constrained to integers as well). The problem is non-linear since several constraints cannot be reduced to linear constraints. 

## Assumptions and Heuristics
Due to the complexity of this optimisation problem, there are certain assumptions/heuristics that must be made for the problem to be computationally feasible. 

1. To limit the search space, we can artificially impose integer constraint on $T_1$, $T_2$, $T_3$ and $T_4$. This converts the problem to a pure integer problem, and makes it easier to apply local search techniques.

## Local Search
Local search starts with a candidate solution and then iteratively moves through neighbouring solutions that also satisfies the constraints. Neighbouring solutions differ from the candidate solution in up to $k$ components (k-opt). If a more optimal neighbour is encountered, it becomes the new candidate and the search repeats. The process stops when no better neighbour could be found, or a time limit has been reached. This, however, only guarantees a local optima (hence the name local search). Therefore, local search is usually repeated with a few random candidate solutions as starting point to obtain a list of local optimas. The best local optima is taken to be a "good enough" approximation of the optimal solution.

While local search does not guarantee an optimal solution, it is frequently able to generate solutions that are close to the optimal solution in a realistic time frame. For example, 2-opt is a form of local search that is used to solve the Travelling Salesman Problem (a NP-hard problem with time complexity $O(n!)$).

### Simulated Annealing
Simulated Annealing (SA) is also a form of local search, but it provides an avenue for the search to "break free" of the local optima. It iteratively picks a random neighbouring solution and decides to adopt the solution with probability $P$. If $P$ crosses a random threshold, SA will adopt the new solution regardless if it is objectively better. This allows us to break free of local optima and increase the odds of finding the optimal solution.

The probability function depends on the cost of the current solution $f(s)$, the cost of the new solution $f(s_{new})$, and the current temperature $T$. $P$ must also satisfy the following conditions:
1. $P$ decreases with $T$ and tends to 0 as $T$ tends to 0
2. $P$ must be greater than 0 even when $f(s_{new}) > f(s)$ - this allows us to escape local minima
3. $P$ increases when $f(s_{new}) - f(s)$ increases

SA starts with a candidate solution $s_0$ and a starting temperature $T_0$. In each iteration, the temperature $T$ decreases, and the algorithm ends when $T$ reaches 0 (no changes can be made since $P$ is 0), or when the maximum number of steps have been encountered. The pseudocode for this can be found on [wikipedia](https://en.wikipedia.org/wiki/Simulated_annealing#Pseudocode).

## Neighbours
### Independent Variables
Given the number of variables that can be tweaked, we need to limit the search space to a feasible region for neighbours. Firstly, $T_2$ is a dependent variable that is directly constrained by $T_1$, $P_S$ and $P_H$. Therefore, it is easier to modify $T_2$ through $T_1$, $P_S$ and $P_H$ than vice versa. The same holds true for $T_4$ and its dependencies $T_3$, $P_H$ and $P_D$. As such, our neighbourhood search should be done by twewaking the values of $T_1$, $T_3$, $P_S$, $P_H$ and $P_D$.

### Operations
Next, we define certain operations to obtain a neighbouring solution:
1. Switching $P_S$, $P_H$ or $P_D$ for a single trip. This affects the arrival/departure time of the vehicle and all personnel aboard for that trip. 
2. Switching a vehicle of a single personnel. Consequentially, this affects the arrival/departure point and time for that personnel. 
3. Changing $T_1$ or $T_3$ of a single trip. This affects the arrival/departure time of the vehicle and all personnel aboard for that trip.

A neighbouring solution can be defined as one that can be obtained from the current solution using up to 3 of the above operations. However, each operation cannot be performed more than once. 


# Clarifications
1. Are the holding points considerably close to the near banks? Is it one holding point per near bank?
2. Is there a hard constraint on the maximum waiting time? This allows us to decide between setting it as an objective or constraint. (set as 10 mins - sai song)
