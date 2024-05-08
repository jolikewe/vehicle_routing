import numpy as np
import pandas
from ortools.sat.python import cp_model

class Model(cp_model.CpModel):
    def __init__(self, params:dict):
        super().__init__()
        self.num_personnel = params['num_personnel']
        self.max_trips = params['max_trips']
        self.num_points_start = params['num_points_start']
        self.points_holding = params['num_points_holding']
        self.num_points_destination = params['num_points_destination']
        self.points_start = []
        self.points_holding = []
        self.points_destination = []
        self.points = []
        self.num_vehicles_alpha = params['num_vehicles_alpha']
        self.num_vehicles_beta = params['num_vehicles_beta']
        self.vehicles_alpha = []
        self.vehicles_beta = []
        self.vehicles = []
        self.vehicle_capacity = params['vehicle_capacity']
        self.loading_time = params['loading_time']
        self.routes = None

        for v in range(self.num_vehicles_alpha):
            self.vehicles_alpha += [f'Va{v+1}']
            self.vehicles += [f'Va{v+1}']
        for v in range(self.num_vehicles_beta):
            self.vehicles_beta += [f'Vb{v+1}']
            self.vehicles += [f'Vb{v+1}']
        for p in range(params['num_points_start']):
            self.points_start += [f'S{p+1}']
            self.points += [f'S{p+1}']
        for p in range(params['num_points_holding']):
            self.points_holding += [f'H{p+1}']
            self.points += [f'H{p+1}']
        for p in range(self.num_points_destination):
            self.points_destination += [f'D{p+1}']
            self.points += [f'D{p+1}']

    def generate_route(self, seed=8):
        np.random.seed(seed=seed)

        self.routes = pandas.DataFrame(999, index=self.points, columns=self.points)
        for s in self.points_start:
            for h in self.points_holding:
                self.routes.loc[s, h] = np.random.randint(5, 12)
        for h in self.points_holding:
            for d in self.points_destination:
                self.routes.loc[h, d] = np.random.randint(5, 12)
        for i in range(self.routes.shape[0]):
            for j in range(self.routes.shape[1]):
                self.routes.iloc[j, i] = self.routes.iloc[i, j]


    #### Variables ####

    def generate_personnel_route(self):
        self.var_x, self.var_y = {}, {}
        for n in range(self.num_personnel):
            for v in self.vehicles_alpha:
                for r in range(self.max_trips):
                    for i in self.points_holding:
                        for j in self.points_destination:
                            self.var_x[(n, v, r, i, j)] = self.NewBoolVar(f'x_{n}{v}{r}{i}{j}')
        for n in range(self.num_personnel):
            for v in self.vehicles_beta:
                for r in range(self.max_trips):
                    for i in self.points_start:
                        for j in self.points_holding:
                            self.var_y[(n, v, r, i, j)] = self.NewBoolVar(f'y_{n}{v}{r}{i}{j}')

    def generate_vehicle_route(self):
        self.var_p, self.var_q = {}, {}
        for v in self.vehicles_alpha:
            for r in range(self.max_trips):
                for i in self.points_holding:
                    for j in self.points_destination:
                        self.var_p[(v, r, i, j)] = self.NewBoolVar(f'p_{v}{r}{i}{j}')
                        self.var_p[(v, r, j, i)] = self.NewBoolVar(f'p_{v}{r}{j}{i}')
        for v in self.vehicles_beta:
            for r in range(self.max_trips):
                for i in self.points_start:
                    for j in self.points_holding:
                        self.var_q[(v, r, i, j)] = self.NewBoolVar(f'q_{v}{r}{i}{j}')
                        self.var_q[(v, r, j, i)] = self.NewBoolVar(f'q_{v}{r}{j}{i}')

    def generate_node_visitation_vehicle(self):
        self.node_v = {}
        for v in self.vehicles_alpha:
            for r in range(self.max_trips):
                for i in self.points_holding + self.points_destination:
                    self.node_v[(v, r, i)] = self.NewBoolVar(f'w_{v}{r}{i}')
        for v in self.vehicles_beta:
            for r in range(self.max_trips):
                for i in self.points_start + self.points_holding:
                    self.node_v[(v, r, i)] = self.NewBoolVar(f'w_{v}{r}{i}')

    def generate_node_visitation_personnel(self):
        self.node_p = {}
        for n in range(self.num_personnel):
            for i in self.points:
                self.node_p[(n, i)] = self.NewBoolVar(f'z_{n}{i}')
        return self.node_p

    def generate_arrival_vehicle(self):
        self.arrival_v = {}
        for v in self.vehicles_alpha:
            for r in range(self.max_trips):
                for i in self.points_holding + self.points_destination:
                    self.arrival_v[(v, r, i)] = self.NewIntVar(0, 10000, f'T_{v}{r}{i}')
        for v in self.vehicles_beta:
            for r in range(self.max_trips):
                for i in self.points_start + self.points_holding:
                    self.arrival_v[(v, r, i)] = self.NewIntVar(0, 10000, f'T_{v}{r}{i}')

    def generate_arrival_personnel(self):
        self.arrival_p = {}
        for n in range(self.num_personnel):
            for i in self.points_holding + self.points_destination:
                self.arrival_p[(n, i)] = self.NewIntVar(0, 10000, f't_{n}{i}')

    def generate_waiting(self):
        self.waiting_v = {}
        for v in self.vehicles_alpha:
            for r in range(self.max_trips):
                for i in self.points:
                    self.waiting_v[(v, r, i)] = self.NewIntVar(0, 30, f'T_{v}{r}{i}')
        for v in self.vehicles_beta:
            for r in range(self.max_trips):
                for i in self.points:
                    self.waiting_v[(v, r, i)] = self.NewIntVar(0, 30, f'T_{v}{r}{i}')


    #### Constraints ####

    ## i. Vehicle arrives and departs from same HOLDING node if node is visited during that trip
    def generate_vehicle_node_constraint(self):
        for v in self.vehicles_alpha:
            for r in range(self.max_trips):
                for i in self.points_holding:
                    depart_from_node = [self.var_p[var] for var in self.var_p if (var[0]==v and var[1]==r and var[2]==i)]
                    arrive_at_node = [self.var_p[var] for var in self.var_p if (var[0]==v and var[1]==r and var[3]==i)]
                    self.Add(sum(depart_from_node) == self.node_v[(v, r, i)])
                    self.Add(sum(arrive_at_node) == self.node_v[(v, r, i)])
        for v in self.vehicles_beta:
            for r in range(self.max_trips):
                for i in self.points_holding:
                    depart_from_node = [self.var_q[var] for var in self.var_q if (var[0]==v and var[1]==r and var[2]==i)]
                    arrive_at_node = [self.var_q[var] for var in self.var_q if (var[0]==v and var[1]==r and var[3]==i)]
                    self.Add(sum(depart_from_node) == self.node_v[(v, r, i)])
                    self.Add(sum(arrive_at_node) == self.node_v[(v, r, i)])
    
    ## ii. Vehicle v departs from node i (START/DESTINATION) in trip r+1, if trip exsits, if it arrives at node i in trip r
    def generate_vehicle_next_trip_constraint(self):
        for v in self.vehicles_alpha:
            for r in range(self.max_trips-1):
                for i in self.points_destination:
                    arrive_at_node = [self.var_p[var] for var in self.var_p if (var[0]==v and var[1]==r and var[3]==i)]
                    depart_from_node = [self.var_p[var] for var in self.var_p if (var[0]==v and var[1]==r+1 and var[-2]==i)]
                    self.Add(sum(arrive_at_node) >= sum(depart_from_node))
                    self.Add(sum(depart_from_node) == self.node_v[(v, r+1, i)])
        for v in self.vehicles_beta:
            for r in range(self.max_trips-1):
                for i in self.points_start:
                    arrive_at_node = [self.var_q[var] for var in self.var_q if (var[0]==v and var[1]==r and var[3]==i)]
                    depart_from_node = [self.var_q[var] for var in self.var_q if (var[0]==v and var[1]==r+1 and var[-2]==i)]
                    self.Add(sum(arrive_at_node) >= sum(depart_from_node))
                    self.Add(sum(depart_from_node) == self.node_v[(v, r+1, i)])

    ## iii. Only 1 start<->holding route / holding<->destination route per trip per vehicle
    def generate_vehicle_trip_limit_constraint(self):
        for v in self.vehicles_alpha:
            for r in range(self.max_trips):
                depart_from_destination = [self.var_p[var] for var in self.var_p if (var[0]==v and var[1]==r and (var[2] in self.points_destination))]
                self.Add(sum(depart_from_destination) <= 1)
                depart_from_holding = [self.var_p[var] for var in self.var_p if (var[0]==v and var[1]==r and (var[2] in self.points_holding))]
                self.Add(sum(depart_from_holding) <= 1)
        for v in self.vehicles_beta:
            for r in range(self.max_trips):
                depart_from_start = [self.var_q[var] for var in self.var_q if (var[0]==v and var[1]==r and (var[2] in self.points_start))]
                self.Add(sum(depart_from_start) <= 1)
                depart_from_holding = [self.var_q[var] for var in self.var_q if (var[0]==v and var[1]==r and (var[2] in self.points_holding))]
                self.Add(sum(depart_from_holding) <= 1)

    ## iv. Personnel arrives and departs from same HOLDING node if node is visited
    def generate_personnel_node_constraint(self):
        for n in range(self.num_personnel):
            for i in self.points_holding:
                arrive_at_node = [self.var_y[var] for var in self.var_y if (var[0]==n and var[-1]==i )]
                depart_from_node = [self.var_x[var] for var in self.var_x if (var[0]==n and var[-2]==i)]
                self.Add(sum(arrive_at_node) == self.node_p[(n, i)])
                self.Add(sum(depart_from_node) == self.node_p[(n, i)])

    ## v. Personnel visits node only if it boards the vehicle
    def generate_board_vehicle_constraint(self):
        for n in range(self.num_personnel):
            for i in self.points_destination:
                arrive_at_node = [var for var in self.var_x if (var[0]==n and var[4]==i)]
                total = 0
                for var in arrive_at_node:
                    xp = self.NewIntVar(0, 10000, 'xp')   ## temporary variable for nonlinear constraint
                    self.AddMultiplicationEquality(xp, self.var_x[var], self.var_p[var[1:]])
                    total += xp
                self.Add(self.node_p[n, i] == total)
        for n in range(self.num_personnel):
            for i in self.points_holding:
                arrive_at_node = [var for var in self.var_y if (var[0]==n and var[4]==i)]
                total = 0
                for var in arrive_at_node:
                    yq = self.NewIntVar(0, 10000, 'yq')   ## temporary variable for nonlinear constraint
                    self.AddMultiplicationEquality(yq, self.var_y[var], self.var_q[var[1:]])
                    total += yq
                self.Add(self.node_p[n, i] == total)

    ## vi. Personnel departs from start node if it is stationed there
    def generate_personnel_start_constraint(self):
        for n in range(self.num_personnel):
            for i in self.points_start:
                depart_from_start = [var for var in self.var_y if (var[0]==n and var[-2]==i)]
                total = 0
                for var in depart_from_start:
                    yq = self.NewIntVar(0, 10000, 'yq')   ## temporary variable for nonlinear constraint
                    self.AddMultiplicationEquality(yq, self.var_y[var], self.var_q[var[1:]])
                    total += yq
                self.Add(self.node_p[n, i] == total)

    ## vii Only 1 start->holding & holding->destination per personnel
    def generate_personnel_trip_limit_constraint(self):
        for n in range(self.num_personnel):
            route_alpha = [self.var_x[var] for var in self.var_x if var[0]==n]
            self.Add(sum(route_alpha) == 1)
            route_beta = [self.var_y[var] for var in self.var_y if var[0]==n]
            self.Add(sum(route_beta) == 1)

    ## viii. Vehicle capacity constraints
    def generate_vehicle_capacity_constraint(self):
        for v in self.vehicles_alpha:
            for r in range(self.max_trips):
                for i in self.points_holding:
                    for j in self.points_destination:
                        arrive_at_node = [var for var in self.var_x if var[1:]==(v, r, i, j)]
                        total = 0
                        for var in arrive_at_node:
                            xp = self.NewIntVar(0, 10000, 'xp')   ## temporary variable for nonlinear constraint
                            self.AddMultiplicationEquality(xp, self.var_x[var], self.var_p[var[1:]])
                            total += xp
                        self.Add(total <= self.vehicle_capacity)
        for v in self.vehicles_beta:
            for r in range(self.max_trips):
                for i in self.points_start:
                    for j in self.points_holding:
                        arrive_at_node = [var for var in self.var_y if var[1:]==(v, r, i, j)]
                        total = 0
                        for var in arrive_at_node:
                            yq = self.NewIntVar(0, 10000, 'yq')   ## temporary variable for nonlinear constraint
                            self.AddMultiplicationEquality(yq, self.var_y[var], self.var_q[var[1:]])
                            total += yq
                        self.Add(total <= self.vehicle_capacity)

    ## ix. Arrival time at START/DESTINATION >= Arrival time at HOLDING + Loading time + Waiting time + 
    ##     Travel time if vehicle v takes route during trip r
    def generate_vehicle_arrival_constraint(self):
        for v in self.vehicles_alpha:
            for r in range(self.max_trips):
                for i in self.points_destination:
                    arrive_at_node = [var for var in self.var_p if (var[0]==v and var[1]==r and var[3]==i)]
                    total = 0
                    for var in arrive_at_node:
                        pT = self.NewIntVar(0, 10000, 'pT')   ## temporary variable for nonlinear constraint
                        self.AddMultiplicationEquality(pT, self.var_p[var], self.arrival_v[var[:-1]])
                        pb = self.NewIntVar(0, 30, 'pb')
                        self.AddMultiplicationEquality(pb, self.var_p[var], self.waiting_v[var[:-1]])
                        total += pT + pb + self.var_p[var] * (self.loading_time + self.routes.loc[var[-2:]])
                    self.Add(self.arrival_v[(v, r, i)] >= total)
        for v in self.vehicles_beta:
            for r in range(self.max_trips):
                for i in self.points_start:
                    arrive_at_node = [var for var in self.var_q if (var[0]==v and var[1]==r and var[3]==i)]
                    total = 0
                    for var in arrive_at_node:
                        qT = self.NewIntVar(0, 10000, 'qT')   ## temporary variable for nonlinear constraint
                        self.AddMultiplicationEquality(qT, self.var_q[var], self.arrival_v[var[:-1]])
                        qb = self.NewIntVar(0, 30, 'qb')
                        self.AddMultiplicationEquality(qb, self.var_q[var], self.waiting_v[var[:-1]])
                        total += qT + qb + self.var_q[var] * (self.loading_time + self.routes.loc[var[-2:]])
                    self.Add(self.arrival_v[(v, r, i)] >= total)

    ## x. Arrival time of trip r+1 of vehicle v at HOLDING >= Arrival time at START/DESTINATION of trip r + 
    ##    Loading time + Waiting time + Travel time if route taken
    def generate_holding_arrival_constraint(self):
        for v in self.vehicles_alpha:
            for r in range(self.max_trips-1):
                for i in self.points_holding:
                    arrive_at_node = [var for var in self.var_p if (var[0]==v and var[1]==r+1 and var[3]==i)]
                    total = 0
                    for var in arrive_at_node:
                        time_var = (var[0], var[1]-1, var[2])
                        pT = self.NewIntVar(0, 10000, 'pT')   ## temporary variable for nonlinear constraint
                        self.AddMultiplicationEquality(pT, self.var_p[var], self.arrival_v[time_var])
                        pb = self.NewIntVar(0, 30, 'pb')
                        self.AddMultiplicationEquality(pb, self.var_p[var], self.waiting_v[var[:-1]])
                        total += pT + pb + self.var_p[var] * (self.loading_time + self.routes.loc[var[-2:]])
                    self.Add(self.arrival_v[(v, r+1, i)] >= total)
        for v in self.vehicles_beta:
            for r in range(self.max_trips-1):
                for i in self.points_holding:
                    arrive_at_node = [var for var in self.var_q if (var[0]==v and var[1]==r+1 and var[3]==i)]
                    total = 0
                    for var in arrive_at_node:
                        time_var = (var[0], var[1]-1, var[2])
                        qT = self.NewIntVar(0, 10000, 'qT')
                        self.AddMultiplicationEquality(qT, self.var_q[var], self.arrival_v[time_var])
                        qb = self.NewIntVar(0, 30, 'qb')
                        self.AddMultiplicationEquality(qb, self.var_q[var], self.waiting_v[var[:-1]])
                        total += qT + qb + self.var_q[var] * (self.loading_time + self.routes.loc[var[-2:]])
                    self.Add(self.arrival_v[(v, r+1, i)] >= total)

    ## xi. Arrival time of trip 0 of vehicle v at HOLDING >= Loading time + Waiting time + Travel time if route taken
    def generate_holding_arrival_0_constraint(self):
        for v in self.vehicles_alpha:
            for i in self.points_holding:
                arrive_at_node = [var for var in self.var_p if (var[0]==v and var[1]==0 and var[3]==i)]
                total = 0
                for var in arrive_at_node:
                    pb = self.NewIntVar(0, 30, 'pb')
                    self.AddMultiplicationEquality(pb, self.var_p[var], self.waiting_v[var[:-1]])
                    total += pb + self.var_p[var] * (self.loading_time + self.routes.loc[var[-2:]])
                self.Add(self.arrival_v[(v, 0, i)] >= total)
        for v in self.vehicles_beta:
            for i in self.points_holding:
                arrive_at_node = [var for var in self.var_q if (var[0]==v and var[1]==0 and var[3]==i)]
                total = 0
                for var in arrive_at_node:
                    qb = self.NewIntVar(0, 30, 'qb')
                    self.AddMultiplicationEquality(qb, self.var_q[var], self.waiting_v[var[:-1]])
                    total += qb + self.var_q[var] * (self.loading_time + self.routes.loc[var[-2:]])
                self.Add(self.arrival_v[(v, 0, i)] >= total)

    ## xii. Arrival time of personnel n at node i = Arrival time of trip r of vehicle v at node i if 
    ##      personnel boards it
    def generate_personnel_arrival_constraint(self):
        for n in range(self.num_personnel):
            for i in self.points_destination:
                arrive_at_node = [var for var in self.var_x if (var[0]==n and var[-1]==i)]
                total = 0
                for var in arrive_at_node:
                    time_var = (var[1], var[2], var[4])
                    xT = self.NewIntVar(0, 10000, 'xT')   ## temporary variable for nonlinear constraint
                    self.AddMultiplicationEquality(xT, self.var_x[var], self.arrival_v[time_var])
                    total += xT
                self.Add(self.arrival_p[(n, i)] == total)
        for n in range(self.num_personnel):
            for i in self.points_holding:
                arrive_at_node = [var for var in self.var_y if (var[0]==n and var[-1]==i)]
                total = 0
                for var in arrive_at_node:
                    time_var = (var[1], var[2], var[4])
                    yT = self.NewIntVar(0, 10000, 'yT')   ## temporary variable for nonlinear constraint
                    self.AddMultiplicationEquality(yT, self.var_y[var], self.arrival_v[time_var])
                    total += yT
                self.Add(self.arrival_p[(n, i)] == total)

    ## xiii. Personnel can only board trip r of alpha vehicle if arrival time of trip r of alpha vehicle 
    ##       at Destination node - time taken for route > arrival time of personnel n at holding node i
    def generate_personnel_holding_constraint(self):
        for n in range(self.num_personnel):
            for i in self.points_holding:
                arrive_at_destination = [var for var in self.var_x if (var[0]==n and var[-2]==i)]
                departure_time = 0
                for var in arrive_at_destination:
                    time_var = (var[1], var[2], var[4])
                    xT = self.NewIntVar(0, 10000, 'xT')
                    self.AddMultiplicationEquality(xT, self.var_x[var], self.arrival_v[time_var])
                    departure_time += xT - self.var_x[var]*self.routes.loc[var[-2:]]
                self.Add(departure_time >= self.arrival_p[n, i])

    ## Vehicle v arrives at node i for the same number of trips that depart from node i (start/destination)
    def generate_vehicle_trip_cycle_constraint(self):
        for v in self.vehicles_alpha:
            for i in self.points_destination:
                depart_from_node = [self.var_p[var] for var in self.var_p if (var[0]==v and var[2]==i)]
                arrive_at_node = [self.var_p[var] for var in self.var_p if (var[0]==v and var[3]==i)]
                visited_node = [self.node_v[var] for var in self.node_v if (var[0]==v and var[2]==i)]
                self.Add(sum(depart_from_node) == sum(visited_node))
                self.Add(sum(arrive_at_node) == sum(visited_node))
        for v in self.vehicles_beta:
            for i in self.points_start:
                depart_from_node = [self.var_q[var] for var in self.var_q if (var[0]==v and var[2]==i)]
                arrive_at_node = [self.var_q[var] for var in self.var_q if (var[0]==v and var[3]==i)]
                visited_node = [self.node_v[var] for var in self.node_v if (var[0]==v and var[2]==i)]
                self.Add(sum(depart_from_node) == sum(visited_node))
                self.Add(sum(arrive_at_node) == sum(visited_node))

    ## Subsequent trips of vehicle r can only take place if prior trip of the same route is taken
    def generate_vehicle_subsequent_trip_constraint(self):
        for v in self.vehicles_alpha:
            for r in range(self.max_trips-1):
                trips = [var for var in self.var_p if (var[0]==v and var[1]==r)]
                for var in trips:
                    temp_var = (v, r+1, var[2], var[3])
                    self.Add(self.var_p[temp_var] <= self.var_p[var])
        for v in self.vehicles_beta:
            for r in range(self.max_trips-1):
                trips = [var for var in self.var_q if (var[0]==v and var[1]==r)]
                for var in trips:
                    temp_var = (v, r+1, var[2], var[3])
                    self.Add(self.var_q[temp_var] <= self.var_q[var])