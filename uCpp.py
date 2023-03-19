import pandas as pd,sys,itertools
Verb = pd.read_csv('data/node1.csv')
Edge = pd.read_csv('data/edge1.csv')
V = len(Verb.index)
E = len(Edge.index)
list_0_verb = [V]
list_0_edge = []
for i in range(0,V):
    list_0_edge.append([i,V,0])
    list_0_edge.append([V,i,0])
columns_0_verb = ["id"]
columns_0_edge = ["node1","node2","weight"]
Edge_reverse = Edge.loc[:, ['node2', 'node1', 'weight']]
Edge_reverse.columns = ['node1','node2','weight']
Edge = Edge.append(Edge_reverse, ignore_index = True)
verb = Verb['id'].tolist()
edge = Edge.to_numpy().tolist()
list_0_verb = pd.DataFrame(data=list_0_verb,columns = columns_0_verb)
list_0_edge = pd.DataFrame(data=list_0_edge,columns = columns_0_edge)
Verb = Verb.append(list_0_verb, ignore_index = True)
Edge = Edge.append(list_0_edge, ignore_index = True)
verb_MBF = Verb.to_numpy().tolist()
edge_MBF = Edge.to_numpy().tolist()

T = [0] * V
for a in range(0,2 * E):
    T[edge[a][0]] += 1
Tj = []
tj = []
i = 0
for j in range(0,V):
    if T[j] % 2 == 1:
        Tj.append([i,j])
        tj.append(i)
        i += 1
def Tjoin(x):
    return Tj[tj.index(x)][1]

#Dijkstra
G = []
Q = []
for s in range(0,V):
    l_Dij = [10 ** 18] * V
    l_Dij[s] = 0
    q_Dij = [0] * V
    R = []
    a = 0
    while 0 == 0:
        lv_Dij = 10 ** 18
        for i in range(0,V):
            if i not in R and lv_Dij >l_Dij[i]:
                lv_Dij = l_Dij[i]
                v_Dij = i
        if v_Dij in R:
            break
        R.append(v_Dij)
        for i in range(0,2 * E):
            if v_Dij == edge[i][0] and edge[i][1] not in R:
                if l_Dij[edge[i][1]] > l_Dij[edge[i][0]] + edge[i][2]:
                    l_Dij[edge[i][1]] = l_Dij[edge[i][0]] + edge[i][2]
                    q_Dij[edge[i][1]] = edge[i][0]
        R.sort()
        if R == verb:
            G.append(l_Dij)
            Q.append(q_Dij)
            break

#MinimumWeightPerfectMatchingProblem
#メトリック閉包Gの辺のリストを作成
V_G = len(Tj)
kv_MWPMP = [0] * V_G
mu_MWPMP = []
rho_MWPMP = []
rho_MWPMP.append(list(range(0,V_G)))
phi_MWPMP = []
phi_MWPMP.append(list(range(0,V_G)))
sigma_MWPMP = []
chi_MWPMP = []
for i in range(0,V_G):
    mu_MWPMP.append(i)
    sigma_MWPMP.append(i)
    chi_MWPMP.append(i)
#verb_Gは1列目：頂点番号，2列目：マーク（偶か奇か），3列目：kv，4列目：mu_MWPMP
#5列目：sigma_MWPMP，6列目：chi_MWPMP，7列目：含まれている極大な花の頂点集合
verb_G = []
for i in range(0,V_G):
    verb_G.append([i,'偶',kv_MWPMP[i],mu_MWPMP[i],sigma_MWPMP[i],chi_MWPMP[i],[i]])
edge_G = []
for i in range(0,V_G):
    for j in range(0,V_G):
        if G[Tjoin(i)][Tjoin(j)] != 0:
            edge_G.append([i,j,G[Tjoin(verb_G[i][0])][Tjoin(verb_G[j][0])]])
E_G = len(edge_G)
#①
K_MWPMP = 0
delta_MWPMP = 0
z_MWPMP = []
for i in range(0,V_G):
    z = float(10 ** 18)
    for j in range(0,E_G):
        if i == edge_G[j][0] and z > edge_G[j][2] / 2:
            z = edge_G[j][2] / 2
    z_MWPMP.append(z)
zeta_MWPMP = z_MWPMP.copy()
#B_MWPMPは1列目：集合，2列目：t^A，3列目：偶花か奇花か森外か，4列目：極大かどうか
#5列目：z，6列目：zeta
B_MWPMP = []
B_mwpmp = []
for i in range(0,V_G):
    B_MWPMP.append([[verb_G[i][0]],10 ** 18,'偶花','極大',z_MWPMP[i],zeta_MWPMP[i]])
    B_mwpmp.append([verb_G[i][0]])
#1列目：下付き文字，2列目：上付き文字（集合），3列目：値
tauv_MWPMP = []
tv_MWPMP = []
tauv_mwpmp = []
tv_mwpmp = []
for i in range(0,V_G):
    tauv_MWPMP.append([])
    tv_MWPMP.append([])
    tauv_mwpmp.append([])
    tv_mwpmp.append([])
    for x in range(0,V_G):
        tauv_MWPMP[i].append([[x],x])
        tv_MWPMP[i].append([[x],G[i][x] - B_MWPMP[B_mwpmp.index([i])][5] + delta_MWPMP])
        tauv_mwpmp[i].append([x])
        tv_mwpmp[i].append([x])

#ーーーーーーーーーーーーーーーーーーーーー
#②
def MWPMP2():
    for i in range(0,len(B_MWPMP)):
        B_MWPMP[i][1] = 10 ** 18

    for v in range(0,V_G):
        if verb_G[v][1] == '偶':
            UPDATE(v)
    MWPMP3()
    return

#③

def MWPMP3():
    global delta_MWPMP
    E1_MWPMP = [10 ** 18]
    E2_MWPMP = [10 ** 18]
    E3_MWPMP = [10 ** 18]
    for A in B_MWPMP:
        if A[2] == '奇花' and A[3] == '極大' and len(A[0]) > 1:
            E1_MWPMP.append(A[4])
        elif A[2] == '森外' and A[3] == '極大':
            E2_MWPMP.append(A[1] - delta_MWPMP -A[5])
        elif A[2] == '偶花' and A[3] == '極大':
            E3_MWPMP.append((A[1] - delta_MWPMP - A[5]) / 2)
    epsilon1 = min(E1_MWPMP)
    epsilon2 = min(E2_MWPMP)
    epsilon3 = min(E3_MWPMP)
    epsilon = min(epsilon1,epsilon2,epsilon3)
    if epsilon == 10 ** 18:
        sys.exit()
    B_tuple = []
    for A in B_mwpmp:
        B_tuple.append(tuple(A))
    for A in B_MWPMP:
        if A[2] == '偶花' and A[3] == '極大':
            B_MWPMP[B_mwpmp.index(A[0])][4] += epsilon
            for i in range(0,len(A[0])):
                AA = set(B_tuple).intersection(set(itertools.combinations(A[0], len(A[0]) - i)))
                for aa in AA:
                    B_MWPMP[B_mwpmp.index(list(aa))][5] += epsilon
        if A[2] == '奇花' and A[3] == '極大':
            B_MWPMP[B_mwpmp.index(A[0])][4] -= epsilon
            for i in range(0,len(A[0])):
                AA = set(B_tuple).intersection(set(itertools.combinations(A[0], len(A[0]) - i)))
                for aa in AA:
                    B_MWPMP[B_mwpmp.index(list(aa))][5] -= epsilon
    delta_MWPMP += epsilon
    MWPMP4(epsilon,epsilon1,epsilon2,epsilon3)

def MWPMP4(epsilon,epsilon1,epsilon2,epsilon3):
    if epsilon == epsilon1:
        MWPMP8()
        return
    if epsilon == epsilon2:
        X = []
        Y = []
        Z = []
        for A in verb_G:
            if A[1] == '偶':
                X.append(A[0])
        for A in B_MWPMP:
            if A[2] == '森外':
                Y.append(A[0])
                if Z == []:
                    Z = A[0].copy()
                else:
                    Z.extend(A[0])
        if X != [] and Y != [] and list(set(X) - set(Z)) != [] and list(set(Z) - set(X)) != []:
            for i in range(0,len(X)):
                for j in range(0,len(Y)):
                    x = X[i % len(X)]
                    A = Y[j % len(Y)]
                    if tv_MWPMP[x][tv_mwpmp[x].index(A)][1] - delta_MWPMP - B_MWPMP[B_mwpmp.index(A)][5] == 0:
                        MWPMP5(x,A[0])
                        return
    if epsilon == epsilon3:
        X = []
        for i in range(0,V_G):
            if verb_G[i][1] == '偶':
                X.append(verb_G[i][0])
        for A in B_MWPMP:
            if A[2] == '偶花' and A[3] == '極大':
                XX = list(set(X).difference(set(A[0])))
                if XX == []:
                    continue
                for i in range(0,len(XX)):
                    x = XX[i]
                    if tv_MWPMP[x][tv_mwpmp[x].index(A[0])][1] - delta_MWPMP - B_MWPMP[B_mwpmp.index(A[0])][5] == 0:
                        Px = TREEPATH(x)
                        y = A[0][0]
                        Py = TREEPATH(y)
                        h_MWPMP = int((len(Px) - 1) / 2)
                        j_MWPMP = int((len(Py) - 1) / 2)
                        if list(set(Px).intersection(set(Py))) == []:
                            MWPMP6(Px,Py,h_MWPMP,j_MWPMP)
                        else:
                            MWPMP7(Px,Py)
                        return

def MWPMP5(x,y):
    verb_G[rho_MWPMP[verb_G[y][2]][y]][4] = y
    verb_G[y][5] = x
    for v in range(0,V_G):
        if rho_MWPMP[verb_G[y][2]][y] == rho_MWPMP[verb_G[v][2]][v]:
            verb_G[v][1] = '奇'
            B_MWPMP[B_mwpmp.index([v])][2] = '奇花'
        if verb_G[rho_MWPMP[verb_G[v][2]][v]][3] == rho_MWPMP[verb_G[y][2]][y]:
            verb_G[v][1] ='偶'
            B_MWPMP[B_mwpmp.index([v])][2] = '偶花'
            UPDATE(v)
    for A in B_MWPMP:
        if verb_G[A[0][0]][1] == '奇':
            A[2] = '奇花'
        elif verb_G[A[0][0]][1] == '偶':
            A[2] = '偶花'
    MWPMP3()
    return

def MWPMP6(Px,Py,h_MWPMP,j_MWPMP):
    h = h_MWPMP
    j = j_MWPMP
    M_MWPMP = []
    for v in range(0,V_G):
        M_MWPMP.append(verb_G[v][3])
    for i in range(0,h):
        verb_G[Px[2 * i + 2]][3] = Px[2 * i + 1]
        verb_G[Px[2 * i + 1]][3] = Px[2 * i + 2]
    for i in range(0,j):
        verb_G[Py[2 * i + 2]][3] = Py[2 * i + 1]
        verb_G[Py[2 * i + 1]][3] = Py[2 * i + 2]
    verb_G[Px[0]][3] = Py[0]
    verb_G[Py[0]][3] = Px[0]
    rho_MWPMP[verb_G[Px[0]][2]][Px[0]] = Px[0]
    rho_MWPMP[verb_G[Py[0]][2]][Py[0]] = Py[0]
    PRU_MWPMP = list(set(Px + Py))
    PRU(PRU_MWPMP,M_MWPMP)
    PRU_MWPMP = []
    for v in range(0,V_G):
        P = TREEPATH(v)
        if Px[2 * h] in P or Py[2 * j] in P:
            for i in P:
                PRU_MWPMP.append(i)
    for i in PRU_MWPMP:
        verb_G[i][1] = '森外'
        B_MWPMP[B_mwpmp.index([i])][2] = '森外'
    for A in B_MWPMP:
        if verb_G[A[0][0]][1] == '森外':
            A[2] = '森外'


    for v in range(0,V_G):
        if v == verb_G[v][3]:
            break
        if v == V_G - 1:
            return
    MWPMP2()
    return

def MWPMP7(Px,Py):
    X = []
    for v in range(0,V_G):
        if verb_G[v][1] != '偶':
            X.append(x)
    global K_MWPMP,rho_MWPMP,phi_MWPMP
    for i in range(0,V_G):
        if verb_G[i][1] == '偶' and i in set(Px).intersection(set(Py)) and rho_MWPMP[verb_G[i][2]][i] == i:
            r = i
            break
    h_prime = int(Px.index(r) / 2)
    j_prime = int(Py.index(r) / 2)
    Pxr = Px[0:2 * h_prime + 1]
    Pyr = Py[0:2 * j_prime + 1]
    A = []
    for i in range(0,V_G):
        if rho_MWPMP[verb_G[i][2]][i] in set(Pxr).union(set(Pyr)):
            A.append(i)
    K_MWPMP += 1
    B_MWPMP.append([A,10 ** 18,'偶花','極大',0,0])
    B_mwpmp.append(A)
    for v in A:
        verb_G[v][6] = A
    for i in B_mwpmp:
        if i != A and sorted(set(A).union(set(i))) == sorted(set(A)):
            B_MWPMP[B_mwpmp.index(i)][3] = '極大ではない'
    #1列目：K，2列目：k_v，3列目：v
    B_prime = []
    for v in A:
        verb_G[v][2] += 1
        B_prime.append([K_MWPMP,verb_G[v][2],v])
        if verb_G[v][2] + 1 > len(rho_MWPMP):
            rho_MWPMP.append(list(range(0,V_G)))
        rho_MWPMP[verb_G[v][2]][v] = r
        if verb_G[v][2] + 1 > len(phi_MWPMP):
            phi_MWPMP.append(list(range(0,V_G)))
        phi_MWPMP[verb_G[v][2]][v] = phi_MWPMP[verb_G[v][2] - 1][v]
    for i in range(1,h_prime + 1):
        if rho_MWPMP[verb_G[Px[2 * i]][2]][Px[2 * i]] == r:
            phi_MWPMP[verb_G[Px[2 * i]][2]][Px[2 * i]] = Px[2 * i - 1]
        if rho_MWPMP[verb_G[Px[2 * i - 1]][2]][Px[2 * i - 1]] == r:
            phi_MWPMP[verb_G[Px[2 * i - 1]][2]][Px[2 * i - 1]] = Px[2 * i]
    for i in range(1,j_prime + 1):
        if rho_MWPMP[verb_G[Py[2 * i]][2]][Py[2 * i]] == r:
            phi_MWPMP[verb_G[Py[2 * i]][2]][Py[2 * i]] = Py[2 * i - 1]
        if rho_MWPMP[verb_G[Py[2 * i - 1]][2]][Py[2 * i - 1]] == r:
            phi_MWPMP[verb_G[Py[2 * i - 1]][2]][Py[2 * i - 1]] = Py[2 * i]
    if rho_MWPMP[verb_G[Px[0]][2]][Px[0]] == r:
        phi_MWPMP[verb_G[Px[0]][2]][Px[0]] = Py[0]
    if rho_MWPMP[verb_G[Py[0]][2]][Py[0]] == r:
        phi_MWPMP[verb_G[Py[0]][2]][Py[0]] = Px[0]
    for v in range(0,V_G):
        if verb_G[v][1] != '偶' or v in A:
            tv_MWPMP[v].append([A,v])
            tauv_MWPMP[v].append([A,v])
            tv_mwpmp[v].append(A)
            tauv_mwpmp[v].append(A)
            continue
        A_prime = MTS(A,v)
        tv_MWPMP[v].append([A,tv_MWPMP[v][tv_mwpmp[v].index(A_prime)][1] - B_MWPMP[B_mwpmp.index(A)][5]])
        tauv_MWPMP[v].append([A,tv_MWPMP[v][tv_mwpmp[v].index(A_prime)][1]])
        tv_mwpmp[v].append(A)
        tauv_mwpmp[v].append(A)
        if verb_G[v][1] == '偶' and list(set(A).union(set([v]))) not in B_MWPMP:
            B_MWPMP[B_mwpmp.index(A)][1] = min(B_MWPMP[B_mwpmp.index(A)][1],tv_MWPMP[v][tv_mwpmp[v].index(A)][1])
    for v in A:
        verb_G[v][1] = '偶'
        B_MWPMP[B_mwpmp.index([v])][2] = '偶花'
    X = list(set(X).difference(set(A)))
    for v in X:
        UPDATE(v)
    MWPMP3()
    return

def MWPMP8():
    for i in range(0,len(B_MWPMP)):
        if B_MWPMP[i][4] == 0 and len(B_MWPMP[i][0]) >= 3 and B_MWPMP[i][2] == '奇花' and B_MWPMP[i][3] == '極大':
            A = B_MWPMP[i].copy()
    v = A[0][0]
    y = verb_G[rho_MWPMP[verb_G[v][2]][v]][4]
    Q = BLOSSOMPATH(y)
    B_mwpmp.pop(B_MWPMP.index(A))
    B_MWPMP.remove(A)
    for A_prime in B_MWPMP:
        for v in A_prime[0]:
            verb_G[v][6] = A_prime[0]
    for i in B_mwpmp:
        if i != A[0] and sorted(set(A[0]).union(set(i))) == sorted(set(A[0])):
            B_MWPMP[B_mwpmp.index(i)][3] = '極大'
    for w in A[0]:
        verb_G[w][2] -= 1
    l = int((len(Q) - 1) / 2) + 1
    for v in A[0]:
        if rho_MWPMP[verb_G[v][2]][v] not in Q:
            verb_G[v][1] = '森外'
            B_MWPMP[B_mwpmp.index([v])][2] = '森外'
    A_prime = []
    for v in A[0]:
        for i in range(1,l):
            if rho_MWPMP[verb_G[v][2]][v] == Q[2 * i - 1]:
                verb_G[v][1] = '偶'
                B_MWPMP[B_mwpmp.index([v])][2] = '偶花'
                A_prime.append(v)
    for B in B_MWPMP:
        if verb_G[B[0][0]][1] == '偶':
            B[2] = '偶花'
        elif verb_G[B[0][0]][1] == '森外':
            B[2] = '森外'
    for v in A[0]:
        if verb_G[v][1] != '偶' or len(verb_G[v][6]) < 3:
            rho_MWPMP[verb_G[v][2]][v] = v
        else:
            for w in verb_G[v][6]:
                if verb_G[w][3] != verb_G[v][6]:
                    rho_MWPMP[verb_G[v][2]][v] = w
        if B_MWPMP[B_mwpmp.index(verb_G[v][6])][2] == '偶花' and rho_MWPMP[verb_G[v][2]][v] != v:
            P = TREEPATH(v)
            for w in P:
                if P.index(w) == 0 and verb_G[w][3] != P[P.index(w) + 1]:
                    phi_MWPMP[verb_G[w][2]][w] = P[P.index(w) + 1]
                elif P.index(w) == len(P) - 1 and verb_G[w][3] != P[P.index(w) - 1]:
                    phi_MWPMP[verb_G[w][2]][w] = P[P.index(w) - 1]
                elif 0 < P.index(w) < len(P) - 1 and verb_G[w][3] != P[P.index(w) - 1]:
                    phi_MWPMP[verb_G[w][2]][w] = P[P.index(w) - 1]
                elif 0 < P.index(w) < len(P) - 1 and verb_G[w][3] != P[P.index(w) + 1]:
                    phi_MWPMP[verb_G[w][2]][w] = P[P.index(w) + 1]
        elif verb_G[v][1] == '奇':
            P = TREEPATH(v)
            for w in P:
                if P.index(w) == 0 and verb_G[w][3] != P[P.index(w) + 1]:
                    phi_MWPMP[verb_G[w][2]][w] = P[P.index(w) + 1]
                elif P.index(w) == len(P) - 1 and verb_G[w][3] != P[P.index(w) - 1]:
                    phi_MWPMP[verb_G[w][2]][w] = P[P.index(w) - 1]
                elif 0 < P.index(w) < len(P) - 1 and verb_G[w][3] != P[P.index(w) - 1]:
                    phi_MWPMP[verb_G[w][2]][w] = P[P.index(w) - 1]
                elif 0 < P.index(w) < len(P) - 1 and verb_G[w][3] != P[P.index(w) + 1]:
                    phi_MWPMP[verb_G[w][2]][w] = P[P.index(w) + 1]
        else:
            phi_MWPMP[verb_G[v][2]][v] = v
    for v in A[0]:
        if (v == Q[len(Q) - 1] or v == Q[0]) and rho_MWPMP[verb_G[v][2] + 1][v] != v:
            continue
        for i in range(0,l):
            if rho_MWPMP[verb_G[v][2]][v] == Q[2 * i]:
                for k in range(0,len(Q)):
                    if rho_MWPMP[verb_G[Q[k]][2]][Q[k]] == rho_MWPMP[verb_G[v][2]][v]:
                        verb_G[rho_MWPMP[verb_G[v][2]][v]][4] = Q[k]
                        verb_G[Q[k]][5] = Q[k - 1]
                        break
    for v in A_prime:
        UPDATE(v)
    MWPMP3()
    return

def BLOSSOMPATH(x0):
    X = [x0]
    h = 0
    B_prime = []
    for i in range(0,len(B_MWPMP)):
        if len(B_MWPMP[i][0]) >= 3 and x0 in B_MWPMP[i][0]:
            B_prime.append(B_MWPMP[i])
    kx0 = verb_G[x0][2]
    while X[2 * h] != rho_MWPMP[kx0][x0]:
        X.append(verb_G[X[2 * h]][3])
        Y = B_prime.copy()
        sorted(Y,key = len)
        for A in B_prime:
            if X[2 * h + 1] in A[0]:
                i = Y.index(A) + 1
                break
        X.append(phi_MWPMP[i][X[2 * h + 1]])
        for A in B_MWPMP:
            if len(A[0]) >= 3 and X[2 * h + 2] in A[0] and X[2 * h + 1] not in A[0]:
                B_prime.append(A[0])
        for A in B_prime:
            if X[2 * h + 2] not in A[0]:
                continue
            if X[2 * h + 2] != verb_G[X[2 * h + 2]][3] and verb_G[X[2 * h + 2]][3] not in A:
                B_prime.remove(A)
        h += 1
    return X

def TREEPATH(v):
    x = v
    P = [v]
    P_mu = []
    while 0 == 0:
        y = rho_MWPMP[verb_G[x][2]][x]
        Q = BLOSSOMPATH(x)
        P += Q
        P = list(sorted(set(P), key = P.index))
        if y == verb_G[y][3] or sorted(P) == sorted(P_mu):
            break
        P_mu = P.copy()
        P.append(y)
        P.append(verb_G[y][3])
        Q = BLOSSOMPATH(verb_G[verb_G[y][3]][4])
        for i in range(0,len(Q)):
            j = len(Q) - i - 1
            P.append(Q[j])
        P.append(verb_G[verb_G[y][3]][4])
        P.append(verb_G[verb_G[verb_G[y][3]][4]][5])
        x = verb_G[verb_G[verb_G[y][3]][4]][5]
    return P

def UPDATE(v):
    for x in range(0,V_G):
        tauv_MWPMP[v][x][1] = x
        tv_MWPMP[v][x][1] = G[Tjoin(v)][Tjoin(x)] - B_MWPMP[B_mwpmp.index([v])][5] + delta_MWPMP
    B_prime = B_MWPMP.copy()
    B_prime.sort(key = len)
    for A in B_prime:
        if len(A[0]) >= 2:
            A_prime = MTS(A[0],v)
            tauv_MWPMP[v][tauv_mwpmp[v].index(A[0])][1] = tauv_MWPMP[v][tauv_mwpmp[v].index(A_prime)][1]
            tv_MWPMP[v][tv_mwpmp[v].index(A[0])][1] = tv_MWPMP[v][tv_mwpmp[v].index(A_prime)][1] - B_MWPMP[B_mwpmp.index(A_prime)][5] + B_MWPMP[B_mwpmp.index(A[0])][5]
    for i in range(0,len(B_MWPMP)):
        if v in B_MWPMP[i][0]:
            continue
        if B_MWPMP[i][2] == '偶花' and B_MWPMP[i][3] != '極大':
            continue
        B_MWPMP[i][1] = min(B_MWPMP[i][1],tv_MWPMP[v][tv_mwpmp[v].index(B_MWPMP[i][0])][1])
    return

def PRU(A,M_MWPMP):
    A_prime = A.copy()
    while A_prime != []:
        if verb_G[A_prime[0]][6] != [A_prime[0]] and verb_G[A_prime[0]][1] == '偶':
            B_prime = verb_G[A_prime[0]][6].copy()
            for w in verb_G[A_prime[0]][6]:
                if verb_G[w][3] not in B_prime:
                    break
            for v in verb_G[A_prime[0]][6]:
                rho_MWPMP[verb_G[v][2]][v] = w
                A_prime.remove(v)
        else:
            rho_MWPMP[verb_G[A[0]][2]][A[0]] = A[0]
            A_prime.pop(0)
    A_prime = A.copy()
    while A_prime != []:
        if rho_MWPMP[verb_G[A_prime[0]][2]][A_prime[0]] != A_prime[0] and M_MWPMP[A_prime[0]] != verb_G[A_prime[0]][3]:
            if M_MWPMP[A_prime[0]] != A_prime[0] and verb_G[A_prime[0]][3] != A_prime[0]:
                phi_MWPMP[verb_G[A_prime[0]][2]][A_prime[0]] = M_MWPMP[A_prime[0]]
            else:
                phi_MWPMP[verb_G[A_prime[0]][2]][A_prime[0]] = rho_MWPMP[verb_G[A_prime[0]][2]][A_prime[0]]
        else:
            phi_MWPMP[verb_G[A_prime[0]][2]][A_prime[0]] = A_prime[0]
        A_prime.pop(0)
    return

def MTS(A,v):
    A_prime = []
    AA = []
    for B_prime in B_MWPMP:
        if len(B_prime[0]) >= len(A):
            continue
        A_prime.append(tuple(B_prime[0]))
    for i in range(1,len(A)):
        AA = list(set(A_prime).intersection(set(itertools.combinations(A, len(A) - i))))
        if AA != []:
            break
    z = 10 ** 18
    Z = []
    for A_prime in AA:
        if tv_MWPMP[v][tv_mwpmp[v].index(list(A_prime))][1] - B_MWPMP[B_mwpmp.index(list(A_prime))][5] < z:
            z = tv_MWPMP[v][tv_mwpmp[v].index(list(A_prime))][1] - B_MWPMP[B_mwpmp.index(list(A_prime))][5]
            Z = list(A_prime).copy()
    return Z

MWPMP2()
M = []
for v in range(0,V_G):
    if Tjoin(v) < Tjoin(verb_G[v][3]):
        M.append([Tjoin(v),Tjoin(verb_G[v][3])])
P = []
for i in range(0,len(M)):
    v,w = M[i]
    p = []
    while 0 == 0:
        p.append([v,Q[w][v]])
        if Q[w][v] == w:
            break
        else:
            v = Q[w][v]
    P.append(p)
R = []
for i in range(0,len(P)):
    for j in range(0,len(P[i])):
        for k in range(0,len(P)):
            if i != k and P[i][j] in P[k]:
                break
            if k == len(P) - 1:
                R.append(P[i][j])

for i in range(0,len(R)):
    print(R[i])