from gurobipy import GRB, Model
import gurobipy as grb
import itertools

def solve_RP1(params, OBJ):

    [a, b, r, alpha, beta] = params
    signs = [-1,1]

    # computing c and d so that:
    # Z_0 = a * Z_0^1 + (1-a) * Z_0^0
    # Z_1 = b * Z_1^1 + (1-b) * Z_1^0
    # Z^0 = c * Z_1^0 + (1-c) * Z_0^0
    # Z^1 = d * Z_1^1 + (1-d) * Z_0^1
    denominator1 = (1-a)*(1-r) + (1-b)*r
    denominator2 = a*(1-r) + b*r
    c = (1-b)*r / denominator1
    d = b*r / denominator2

    # create Gurobi model
    m = Model("fair-guarantee")

    # create variables
    # Z00, Z01, Z10, Z11 corresond to Z_0^0, Z_0^1, Z_1^0, Z_1^1
    Z00 = m.addVars(2, 2, 2, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name='Z_0^0')
    Z01 = m.addVars(2, 2, 2, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name='Z_0^1')
    Z10 = m.addVars(2, 2, 2, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name='Z_1^0')
    Z11 = m.addVars(2, 2, 2, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name='Z_1^1')

    # set objective
    if OBJ in ['DOpp_0', 'DOpp_1']:
        i1 = int(OBJ[-1])
        obj = (-1)**i1 * grb.quicksum([
                Z11[i,j,1] - Z01[i,j,1]
                for i, j in itertools.product(range(2),range(2))
        ])
    elif OBJ in ['DR_0', 'DR_1']:
        i1 = int(OBJ[-1])
        obj = (-1)**i1 * grb.quicksum([
                Z10[i,j,1] - Z00[i,j,1]
                for i, j in itertools.product(range(2),range(2))
        ])
    elif OBJ in ['DOdds_00', 'DOdds_01', 'DOdds_10', 'DOdds_11']:
        i1 = int(OBJ[-2])
        i2 = int(OBJ[-1])
        DOpp = (-1)**i1 * grb.quicksum([
                Z11[i,j,1] - Z01[i,j,1]
                for i, j in itertools.product(range(2),range(2))
        ])
        DR = (-1)**i2 * grb.quicksum([
                Z10[i,j,1] - Z00[i,j,1]
                for i, j in itertools.product(range(2),range(2))
        ])
        obj = 0.5 * (DOpp + DR)
    else:
        raise ValueError(f'{OBJ} unrecognized!')
    m.setObjective(obj, GRB.MAXIMIZE)

    # create equality constraints
    Z00_sum = grb.quicksum([
        Z00[i,j,k]
        for i, j, k in itertools.product(range(2),range(2),range(2))
    ])
    Z01_sum = grb.quicksum([
        Z01[i,j,k]
        for i, j, k in itertools.product(range(2),range(2),range(2))
    ])
    Z10_sum = grb.quicksum([
        Z10[i,j,k]
        for i, j, k in itertools.product(range(2),range(2),range(2))
    ])
    Z11_sum = grb.quicksum([
        Z11[i,j,k]
        for i, j, k in itertools.product(range(2),range(2),range(2))
    ])
    sum_list = [Z00_sum, Z01_sum, Z10_sum, Z11_sum]
    m.addConstrs(
        (sum_list[idx] == 1 for idx in range(4)),
        name="equality_1"
    )

    # create absolute value constraint
    # compute Z_1 - Z_0
    fair_abs = [
        [
            [
                (1-b) * Z10[i,j,k] + b * Z11[i,j,k] - (1-a) * Z00[i,j,k] - a * Z01[i,j,k]
                for k in range(2)
            ]
            for j in range(2)
        ]
        for i in range(2)
    ]
    # compute Z^1 - Z^0
    dis_abs = [
        [
            [
                (1-d) * Z01[i,j,k] + d * Z11[i,j,k] - (1-c) * Z00[i,j,k] - c * Z10[i,j,k]
                for k in range(2)
            ]
            for j in range(2)
        ]
        for i in range(2)
    ]

    m.addConstrs(
        (
            signs[i] * fair_abs[i][j][k] >= 0
            for i,j,k in itertools.product(range(2),range(2),range(2))
        ),
        name="fair_abs"
    )
    m.addConstrs(
        (
            signs[j] * dis_abs[i][j][k] >= 0
            for i,j,k in itertools.product(range(2),range(2),range(2))
        ),
        name="dis_abs"
    )
        

    # create fairness constraint
    fair = grb.quicksum(
        [
            fair_abs[1][j][k]
            for j,k in itertools.product(range(2),range(2))
        ]
    )
    m.addConstr(
        fair,
        GRB.LESS_EQUAL,
        alpha,
        "fair"
    )

    # create discriminative constraint
    dis = grb.quicksum(
        [
            dis_abs[i][1][k]
            for i,k in itertools.product(range(2),range(2))
        ]
    )
    m.addConstr(
        dis,
        GRB.GREATER_EQUAL,
        (1 - beta),
        "discriminative"
    )
    m.optimize()

    if m.status == GRB.OPTIMAL:
        return m.objVal
    elif m.status == GRB.INFEASIBLE:
        return np.NaN
    else:
        raise ValueError('!!!')


def solve_RP2(params, OBJ):

    [a, b, r, alpha, beta] = params
    signs = [-1,1]

    # computing c and d so that:
    # Z_0 = a * Z_0^1 + (1-a) * Z_0^0
    # Z_1 = b * Z_1^1 + (1-b) * Z_1^0
    # Z^0 = c * Z_1^0 + (1-c) * Z_0^0
    # Z^1 = d * Z_1^1 + (1-d) * Z_0^1
    denominator1 = (1-a)*(1-r) + (1-b)*r
    denominator2 = a*(1-r) + b*r
    c = (1-b)*r / denominator1
    d = b*r / denominator2

    # create Gurobi model
    m = Model("fair-guarantee")

    # create variables
    # Z00, Z01, Z10, Z11 corresond to Z_0^0, Z_0^1, Z_1^0, Z_1^1
    Z00 = m.addVars(2, 2, 2, 2, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name='Z_0^0')
    Z01 = m.addVars(2, 2, 2, 2, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name='Z_0^1')
    Z10 = m.addVars(2, 2, 2, 2, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name='Z_1^0')
    Z11 = m.addVars(2, 2, 2, 2, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name='Z_1^1')

    # set objective
    if OBJ == 'DPC':
        obj = 0.5 * grb.quicksum([
                signs[l] * (b * Z11[i,j,k,l] - a * Z01[i,j,k,l])
                for i, j ,k, l in itertools.product(range(2),range(2),range(2),range(2))
        ])
    elif OBJ == 'DNC':
        obj = 0.5 * grb.quicksum([
                signs[k] * ((1-b) * Z10[i,j,k,l] - (1-a) * Z00[i,j,k,l])
                for i, j, k, l in itertools.product(range(2),range(2),range(2),range(2))
        ])
    elif OBJ == 'DC':
        DPC = 0.5 * grb.quicksum([
                signs[l] * (b * Z11[i,j,k,l] - a * Z01[i,j,k,l])
                for i, j ,k, l in itertools.product(range(2),range(2),range(2),range(2))
        ])
        DNC = 0.5 * grb.quicksum([
                signs[k] * ((1-b) * Z10[i,j,k,l] - (1-a) * Z00[i,j,k,l])
                for i, j, k, l in itertools.product(range(2),range(2),range(2),range(2))
        ])
        obj = 0.5 * (DPC + DNC)
    else:
        raise ValueError(f'{OBJ} unrecognized!')
    m.setObjective(obj, GRB.MAXIMIZE)

    # create equality constraints
    Z00_sum = grb.quicksum([
        Z00[i,j,k,l]
        for i, j, k, l in itertools.product(range(2),range(2),range(2),range(2))
    ])
    Z01_sum = grb.quicksum([
        Z01[i,j,k,l]
        for i, j, k, l in itertools.product(range(2),range(2),range(2),range(2))
    ])
    Z10_sum = grb.quicksum([
        Z10[i,j,k,l]
        for i, j, k, l in itertools.product(range(2),range(2),range(2),range(2))
    ])
    Z11_sum = grb.quicksum([
        Z11[i,j,k,l]
        for i, j, k, l in itertools.product(range(2),range(2),range(2),range(2))
    ])
    sum_list = [Z00_sum, Z01_sum, Z10_sum, Z11_sum]
    m.addConstrs(
        (sum_list[idx] == 1 for idx in range(4)),
        name="equality_1"
    )

    # create absolute value constraint
    # compute Z_1 - Z_0
    fair_abs = [
        [
            [
                [
                    (1-b) * Z10[i,j,k,l] + b * Z11[i,j,k,l] - (1-a) * Z00[i,j,k,l] - a * Z01[i,j,k,l]
                    for l in range(2)
                ]
                for k in range(2)
            ]
            for j in range(2)
        ]
        for i in range(2)
    ]
    # compute Z^1 - Z^0
    dis_abs = [
        [
            [
                [
                    (1-d) * Z01[i,j,k,l] + d * Z11[i,j,k,l] - (1-c) * Z00[i,j,k,l] - c * Z10[i,j,k,l]
                    for l in range(2)
                ]
                for k in range(2)
            ]
            for j in range(2)
        ]
        for i in range(2)
    ]

    m.addConstrs(
        (
            signs[i] * fair_abs[i][j][k][l] >= 0
            for i,j,k,l in itertools.product(range(2),range(2),range(2),range(2))
        ),
        name="fair_abs"
    )
    m.addConstrs(
        (
            signs[j] * dis_abs[i][j][k][l] >= 0
            for i,j,k,l in itertools.product(range(2),range(2),range(2),range(2))
        ),
        name="dis_abs"
    )
        

    # create fairness constraint
    fair = grb.quicksum(
        [
            fair_abs[1][j][k][l]
            for j,k,l in itertools.product(range(2),range(2),range(2))
        ]
    )
    m.addConstr(
        fair,
        GRB.LESS_EQUAL,
        alpha,
        "fair"
    )

    # create discriminative constraint
    dis = grb.quicksum(
        [
            dis_abs[i][1][k][l]
            for i,k,l in itertools.product(range(2),range(2),range(2))
        ]
    )
    m.addConstr(
        dis,
        GRB.GREATER_EQUAL,
        (1 - beta),
        "discriminative"
    )

    subfair1_abs = [
        [
            [
                [
                    (1-b) * Z10[i,j,k,l] - (1-a) * Z00[i,j,k,l]
                    for l in range(2)
                ]
                for k in range(2)
            ]
            for j in range(2)
        ]
        for i in range(2)
    ]
    subfair2_abs = [
        [
            [
                [
                    b * Z11[i,j,k,l] - a * Z01[i,j,k,l]
                    for l in range(2)
                ]
                for k in range(2)
            ]
            for j in range(2)
        ]
        for i in range(2)
    ]

    m.addConstrs(
        (
            signs[k] * subfair1_abs[i][j][k][l] >= 0
            for i,j,k,l in itertools.product(range(2),range(2),range(2),range(2))
        ),
        name="subfair1_abs"
    )
    m.addConstrs(
        (
            signs[l] * subfair2_abs[i][j][k][l] >= 0
            for i,j,k,l in itertools.product(range(2),range(2),range(2),range(2))
        ),
        name="subfair2_abs"
    )
    m.optimize()

    if m.status == GRB.OPTIMAL:
        return m.objVal
    elif m.status == GRB.INFEASIBLE:
        raise ValueError(f"Under base rates a={a}, b={b}, r={r}, no representation can be both {alpha}-fair and {beta}-discriminative!")
    else:
        return ValueError("An unexpected outcome returned from Gurobi!")


def find_fair_guarantee(params, OBJ):
    """
    Args:
        params: [a, b, r, alpha, beta]
            a, b, r are the base rates
            alpha, beta are the representation's fairness and discriminativeness coefficients
        OBJ: the fairness notion, one of "DOpp", "DR", "DOdds", "DPC", "DNC", "DC"
            DOpp  -> Disparity of Opportunity
            DR    -> Disparity of Regret
            DOdds -> Disparity of Odds
            DPC   -> Disparity of Positive Calibration
            DNC   -> Disparity of Negative Calibration
            DC    -> Disparity of Calibration
    
    Returns:
        fairness guarantee
    """
    if OBJ in ['DOpp', 'DR']:
        out_0 = solve_RP1(params, OBJ + '_0')
        out_1 = solve_RP1(params, OBJ + '_1')
        return max(out_0, out_1)
    elif OBJ == 'DOdds':
        out_00 = solve_RP1(params, OBJ + '_00')
        out_01 = solve_RP1(params, OBJ + '_01')
        out_10 = solve_RP1(params, OBJ + '_10')
        out_11 = solve_RP1(params, OBJ + '_11')
        return max(out_00, out_01, out_10, out_11)
    elif OBJ in ['DPC', 'DNC', 'DC']:
        return solve_RP2(params, OBJ)
    else:
        raise ValueError(f"{OBJ} is not one of the fairness notions!")

if __name__ == '__main__':
    a = .6
    b = .4
    r = 0.5
    alpha = .1
    beta = .1
    params = [a, b, r, alpha, beta]
    max_DOpp = find_fair_guarantee(params, 'DOpp')
    max_DR = find_fair_guarantee(params, 'DR')
    max_DOdds = find_fair_guarantee(params, 'DOdds')
    max_DPC = find_fair_guarantee(params, 'DPC')
    max_DNC = find_fair_guarantee(params, 'DNC')
    max_DC = find_fair_guarantee(params, 'DC')
    print(f'''For base rates a={a}, b={b}, r={r}, 
    using representations that are {alpha}-fair and {beta}-discriminative, 
    we have the following fairness guarantees:
    DOpp <= {max_DOpp:.3f},
    DR <= {max_DR:.3f},
    DOdds <= {max_DOdds:.3f},
    DPC <= {max_DPC:.3f},
    DNC <= {max_DNC:.3f},
    DC <= {max_DC:.3f}.''')