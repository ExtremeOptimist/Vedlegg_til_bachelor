import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
# graph tools


class Vertex:
    """ Punktklasse for å ha kontroll på alle punktene.
    Vil ha kontroll på index i orginalliste, koordinater og vinkel"""
    # __idx = -1

    def __init__(self, idx):
        # Vertex.__idx += 1
        self.idx = idx
        self.x = None
        self.y = None
        self.neighbor_idxs = []
        self.neighbors = []
        self.nr_of_neighbors = 0
        self.angles = []
        self.left = False
        self.category = ''
        self.branch = None
        self.both = False

    def get_coords(self):
        return [self.x, self.y]

    def get_coords_array(self):
        return np.array([self.x, self.y])

    def add_neighbor(self, new_neighbor):
        # legger til en vertex som nabo
        self.neighbors.append(new_neighbor)
        self.nr_of_neighbors += 1

    def add_neighbor_idx(self, new_nb_idx):
        self.neighbor_idxs.append(new_nb_idx)

    # def get_neighbor_angles(self):
    #     self.angles = calc_all_angles(self.neighbors)
    #     return self.angles

    def add_coords(self, punkt):
        self.x = punkt[0]
        self.y = punkt[1]

    def set_branch(self, branch):
        self.branch = branch

    def other_end(self):
        # returnere punktet som er i andre enden av grenen
        # som punktet er en del av.
        # må da spørre gre
        out = self.branch.get_other_end(self)
        return out

    def __key(self):
        return (self.category, self.idx)

    def __hash__(self):
        """Overrides the default implementation"""
        return hash(self.__key())

    def __eq__(self, other):
        if isinstance(other, Vertex):
            return self.__key() == other.__key()
        return NotImplemented

    def __repr__(self):
        return f'{self.idx} med ({self.x}, {self.y}), ant. naboer: {len(self.neighbors)}. '


class Edge:
    __id = -1

    def __init__(self, spidx, epidx):
        Edge.__id += 1
        self.id = Edge.__id
        self.spidx = spidx
        self.epidx = epidx
        self.svtx = None
        self.evtx = None
        self.verticies = []

    def add_vertex(self, v1):
        self.verticies.append(v1)

    def __repr__(self):
        return f'edge {self.id}'

    def __str__(self):
        return self.__repr__()


class Branch:
    __idx = -1

    def __init__(self, start_point):
        Branch.__idx += 1
        self.idx = Branch.__idx
        self.points = [start_point]  # starter med ett punkt
        self.category = ''
        self.age = ''
        self.diff = None
        # retning til grenen i grader mellom
        # høyeste nabo punkt og andre ende.
        self.direction = None

    def get_end_point(self):
        return self.points[-1]

    def __repr__(self):
        return f'{self.idx}'

    def __str__(self):
        return self.__repr__()

    def fill_points(self, nex_nbr):
        point = self.points[0]
        point.set_branch(self)
        next_point = point.neighbors[nex_nbr]
        # self.points.append(next_point)
        counter = 1
        while len(next_point.neighbors) == 2:
            counter += 1
            self.points.append(next_point)
            next_point.set_branch(self)
            # angle_diff = abs(next_point.angles[2] - next_point.angles[1])
            # if 160 < angle_diff < 200:
            if next_point.neighbors[0] not in self.points:
                next_point = next_point.neighbors[0]
            else:
                next_point = next_point.neighbors[1]
            #print(counter)
        self.points.append(next_point)
        next_point.set_branch(self)
        return self.points

    def get_line(self):
        # for plotting, gir linje mellom
        # alle punktene
        xs = []
        ys = []
        for point in self.points:
            xs.append(point.x)
            ys.append(point.y)
        return np.stack((xs, ys), axis=1)

    def get_length(self):
        length = 0
        for x in range(0, len(self.points)-1):
            # tar alle utenom den siste
            new_length = calc_distance(self.points[x].get_coords_array(), self.points[x+1].get_coords_array())
            length = length + new_length
        return length

    def is_center(self):
        if (self.points[0].nr_of_neighbors == 4) or (self.points[-1].nr_of_neighbors == 4):
            out = True
        else:
            out = False
        return out

    def check_left(self):
        if self.points[0].left or self.points[-1].left:
            self.category = 'left'

    def get_most_neighbors_point(self):
        current_point = self.points[0]
        for point in self.points:
            if point.nr_of_neighbors > current_point.nr_of_neighbors:
                current_point = point
        return current_point

    def get_least_neighbors_point(self):
        current_point = self.points[0]
        for point in self.points:
            if point.nr_of_neighbors < current_point.nr_of_neighbors:
                current_point = point
        return current_point

    def check_direction(self, utgpnkt):
        p1 = self.points[0]
        p2 = self.points[-1]
        p3 = utgpnkt
        if p1 == p3:
            end = p2
        else:
            end = p1
        out = calc_angle(p3, end)
        # if 160 < out < 200:
        #     end.category = 'B'
        # kommentert ut fordi dette kan føre til at andre punkter blir B
        # som ikke skal.
        self.direction = out
        return out

    def categorize(self):
        # Vil helst at kategorien skal være sluttpunkt kategori
        # pluss endepunkt kategori
        str1 = self.points[0].category
        str2 = self.points[-1].category
        cat_unsort = str1 + str2
        cat_sort = sorted(cat_unsort)
        # Over sorteres slik at AB og BA blir samme kategori,
        # ved at BA blir omstokket til AB, dette gjør derimot alle andre grenr vanskelige å tolke.
        category = ''.join(cat_sort)
        self.category = category
        return category

    def get_other_end(self, point):
        # Tar inn et endepunkt og
        # returnerer andre enden
        e1 = self.points[0]
        e2 = self.points[-1]
        if point == e1:
            out = e2
        else:
            out = e1
        return out


def calc_distance(p1, p2):
    return (sum((p1-p2)**2))**0.5


def calc_angle(p1, p2):
    x = p2.x - p1.x
    y = p2.y - p1.y
    if x < 0 or (x < 0 and y < 0):
        rad = np.arctan(y/(x+1e-9)) + np.pi
    else:
        rad = np.arctan(y / (x + 1e-9))
    deg = np.rad2deg(rad)
    return deg


# Pixel verdi uthenting:
def pixels_to_datapoints(img):
    # Kan brukes til å bestemme grunnlag for k-means pixler,
    # Altså hvilke piksler som skal være start center.
    # tar inn gråskala bilde
    img_scaled = img
    x_liste = []
    y_liste = []
    y_count = 0
    # centroids = []  # dersom jeg vil ha bestemt mellomrom mellom de tilfeldige clusterene
    for y in img_scaled:
        y_count += 1
        x_count = 0
        for x in y:
            x_count += 1
            # if (x_count % 10 == 0) and (y_count % 10 == 0):
            #   centroids.append([x_count, y_count])
            if x > 10:
                x_liste.append(x_count)
                y_liste.append(y_count * (-1) + img_scaled.shape[0])  # gange for å snu, pluss for forskyv
                # print(f'random pixel verdi = {x}, coord = {x_count, y_count}')

    # Bygger opp lik form som centers, bare uten null pixler:
    relevant_pixels = np.column_stack((x_liste, y_liste))
    return relevant_pixels


# initialierser kmeans parametere
def get_kmeans_centers(datapoints, nr_of_clusters):
    kmeans = KMeans(
        init="k-means++",  # var k-means++
        n_clusters=nr_of_clusters,  # økt fra 60 til 120
        n_init=10,
        max_iter=10,
        random_state=13)  # var 10
    kmeans.fit(datapoints)
    return kmeans.cluster_centers_


# ========================================================================
# ========================================================================
# Fra https://www.codespeedy.com/boruvkas-algorithm-for-minimum-spanning-tree-in-python/
# Tree of a given connected, undirected and weighted graph
# Class to represent a graph
#
class Graph:
    def __init__(self, vertices):
        self.V = vertices  # No. of vertices
        self.graph = []  # default dictionary to store graph

    # function to add an edge to graph
    def add_edge(self, u, v, w):
        self.graph.append([u, v, w])
        # A utility function to find set of an element i
        # (uses path compression technique)

    def find(self, parent, i):
        if parent[i] == i:
            return i
        return self.find(parent, parent[i])

    def union(self, parent, rank, x, y):
        # A function that does union of two sets of x and y
        # (uses union by rank)
        xroot = self.find(parent, x)
        yroot = self.find(parent, y)
        # Attach smaller rank tree under root of high rank tree
        # (Union by Rank)
        if rank[xroot] < rank[yroot]:
            parent[xroot] = yroot
        elif rank[xroot] > rank[yroot]:
            parent[yroot] = xroot
            # If ranks are same, then make one as root and increment
            # its rank by one
        else:
            parent[yroot] = xroot
            rank[xroot] += 1

    # The main function to construct MST using Kruskal's algorithm
    def boruvka_mst(self):
        mst_points = []
        parent = []
        rank = []
        # An array to store index of the cheapest edge of
        # subset. It store [u,v,w] for each component
        cheapest = []
        # Initially there are V different trees.
        # Finally there will be one tree that will be MST
        num_trees = self.V
        mst_weight = 0
        # Create V subsets with single elements
        for node in range(self.V):
            parent.append(node)
            rank.append(0)
            cheapest = [-1] * self.V
        # Keep combining components (or sets) until all
        # compnentes are not combined into single MST
        while num_trees > 1:
            # Traverse through all edges and update
            # cheapest of every component
            for i in range(len(self.graph)):
                # Find components (or sets) of two corners
                # of current edge
                u, v, w = self.graph[i]
                set1 = self.find(parent, u)
                set2 = self.find(parent, v)
                # If two corners of current edge belong to
                # same set, ignore current edge. Else check if
                # current edge is closer to previous
                # cheapest edges of set1 and set2
                if set1 != set2:
                    if cheapest[set1] == -1 or cheapest[set1][2] > w:
                        cheapest[set1] = [u, v, w]
                    if cheapest[set2] == -1 or cheapest[set2][2] > w:
                        cheapest[set2] = [u, v, w]
                        # Consider the above picked cheapest edges and add them
                        # to MST
            for node in range(self.V):
                # Check if cheapest for current set exists
                if cheapest[node] != -1:
                    u, v, w = cheapest[node]
                    set1 = self.find(parent, u)
                    set2 = self.find(parent, v)
                    if set1 != set2:
                        mst_weight += w
                        self.union(parent, rank, set1, set2)
                        # print("Edge %d-%d with weight %d included in MST" % (u, v, w))
                        mst_points.append([u, v])
                        num_trees = num_trees - 1
            # reset cheapest array
            cheapest = [-1] * self.V
        # print("Weight of MST is %d" % mst_weight)
        return mst_points


# =========================================================================
# =========================================================================

def make_mst(points):
    # Bruker nærmeste nabo til å lage en graf der hvert punkt har
    # 5 naboer. Hver nabo har en vekt som er det samme som avstanden.
    nr_of_vertix = len(points)
    nr_of_edges = nr_of_vertix
    # Finner de fem nærmeste naboene til hvert punkt
    g = Graph(nr_of_edges)
    nbrs = NearestNeighbors(n_neighbors=5, algorithm='kd_tree')
    # antall naboer byttet fra 5 til 2
    nbrs.fit(points)
    dist, ind = nbrs.kneighbors(points)
    # Lager kanter til alle nærmeste naboer:
    # srcs = []  # uten Boruvkas algorithm
    # dsts = []  # uten Boruvkas algorithm
    print('Starter for inni make_mst')
    counter = 0
    c_inner = 0
    plz_break = False
    for dist1, pnkt in zip(*[dist, ind]):
        counter += 1
        for wgt, nabo in zip(*[dist1, pnkt]):
            c_inner += 1
            src = pnkt[0]
            dst = nabo
            if src == dst:
                # print(f'Counter inner : {counter}')
                continue
            # else:  # uten Boruvkas algorithm
                # srcs.append(src)  # uten Boruvkas algorithm
                # dsts.append(dst)  # uten Boruvkas algorithm
                # break  # uten Boruvkas algorithm
            g.add_edge(src, nabo, wgt)
            #print(f'Counter inner : {counter}')
            # print('Ferdig med inderste for')
            if counter >= 150:
                print('Breaker på grunn av ferdig')
                plz_break = True
                break
        if plz_break:
            print('Breaker ytterste')
            break
        #print(f'Counter : {counter}')
    mst_points = g.boruvka_mst()
    # mst_points = np.stack((srcs, dsts), axis=1)  # uten Boruvkas algorithm
    # print('Ferdig med yterste for')
    return mst_points


def filter_points(verticies, points, count):
    # Returnerer hvor mange ganger man må filtrere
    # Vil fjerne punkter som har tre nabo ved siden av tre nabo:
    # Fjerner bare ett punkt, og returnerer ny liste,
    # for å finne nye naboer med de nye punktene
    counter = 0
    new_points = points
    break2 = False
    for vertex in verticies:
        if vertex.nr_of_neighbors == 3:
            neighbors = vertex.neighbors
            for neighbor in neighbors:
                if (neighbor.nr_of_neighbors == 3) or (neighbor.nr_of_neighbors == 1):
                    if count:
                        counter += 1
                    else:
                        new_points = np.delete(new_points, neighbor.idx, axis=0)
                        break2 = True
                        break
            if break2:
                break
    return new_points, counter


def make_vertecies_with_neighbors(adj_list):
    # Legger ikke til koordinater, det gjør jeg til slutt
    verticies = []
    done_points = []
    # ===========================================================================
    # bra laging av edger og vertex
    for src, dst in adj_list:
        if src not in done_points:
            # Gjør punkt om til en vertex
            new_vertex = Vertex(src)
            done_points.append(src)
            verticies.append(new_vertex)
        if dst not in done_points:
            # Gjør punkt om til en vertex
            new_vertex = Vertex(dst)
            done_points.append(dst)
            verticies.append(new_vertex)
    # Sorterer slik at vertex id og punkt id er likt:
    verticies.sort(key=lambda vtx: vtx.idx, reverse=False)
    # Vil legge til naboer.
    # Virker rett hvis indexene i punktlisten tilsvarer
    # indexene i vertex listen
    for src, dst in adj_list:
        # Loop for å fylle inn naboer
        pdst = verticies[dst]
        psrc = verticies[src]
        if src not in verticies[dst].neighbor_idxs:
            verticies[dst].add_neighbor_idx(src)
            verticies[dst].add_neighbor(psrc)
        if dst not in verticies[src].neighbor_idxs:
            verticies[src].add_neighbor_idx(dst)
            verticies[src].add_neighbor(pdst)
    return verticies


def make_branches(points):
    # Lage greiner
    # først finne alle endepunter og lage grener av dem
    done_points = []
    branches = []
    for point in points:
        # Loop for å fylle fra alle endepunkter,
        # hvert endepunkt er en gren.
        if len(point.neighbors) == 1:
            branch = Branch(point)
            new_done_points = branch.fill_points(nex_nbr=0)
            branches.append(branch)
            done_points = done_points + new_done_points
    # gå gjennom alle punktene, sjekk om naboen er en del gjort, dersom en nabo ikke er gjort,
    # lag ny gren, men fill fra direction til nabo.
    for point in points:
        neighbors = point.neighbors
        cnt = 0
        if len(neighbors) > 2:
            for neighbor in neighbors:
                if neighbor in done_points:
                    cnt += 1
                    continue
                branch = Branch(point)
                new_done_points = branch.fill_points(cnt)  # punktene som blir fylt ut av neste gren
                branches.append(branch)
                done_points = done_points + new_done_points
                cnt += 1
    return branches


def plot_mst(indexes, points, title):
    plt.title(title)
    for p1, p2 in indexes:
        plt.plot([points[p1, 0], points[p2, 0]], [points[p1, 1], points[p2, 1]], 'r')

    # Finne ut om det er endepunkt:
    point_idx, nr_of_edges = np.unique(indexes, return_counts=True)
    for idx, nr in zip(*[point_idx, nr_of_edges]):
        if nr == 1:
            # endepunkt
            plt.scatter(points[idx, 0], points[idx, 1], s=160, facecolors='none', edgecolors='g')
        if nr == 3:
            # t kryss punkt
            plt.scatter(points[idx, 0], points[idx, 1], s=160, facecolors='none', edgecolors=(0.1, 0.1, 0.1))
    # print(f'uniq: {point_idx}, cnts: {nr_of_edges}')
    plt.scatter(points[:, 0], points[:, 1])


def find_key_verticies(verticies):
    key_verticies = []
    #key_x = []
    #key_y = []
    for vertex in verticies:
        if (vertex.nr_of_neighbors == 1) or (vertex.nr_of_neighbors > 2):
            key_verticies.append(vertex)
    # key_verticies.sort(key=lambda vtx: vtx.category, reverse=False)
    # for vertex in key_verticies:
    #     key_x.append(vertex.x)
    #     key_y.append(vertex.y)
    #key_points = np.stack((key_x, key_y), axis=1)
    return key_verticies


def find_same_keypoints(verts1, verts2):
    # tar inn to lister, og returnerer dem
    # sortert etter kategori
    # set1 = set(verts1)
    # set2 = set(verts2)
    # set3 = set1.intersection(set2)
    new_list1 = []
    new_list2 = []
    for vert1 in verts1:
        # vert1.age = 'old'
        for vert2 in verts2:
            # vert2.age = 'new'
            if vert1.category == vert2.category:
                new_list1.append(vert1)
                new_list2.append(vert2)
    # list1 = list(filter(lambda vt: vt.both, set1))
    # list2 = list(filter(lambda vt: vt.both, set2))
    return new_list1, new_list2


def find_cor_points(list_1, list_2):
    # Denne funksjonen tar inn to lister (før og etter)
    # og returnerer punkter i (2 x ant punkt.) format
    # sortert etter kategori
    list_1.sort(key=lambda vtx: vtx.category, reverse=False)
    list_2.sort(key=lambda vtx: vtx.category, reverse=False)
    for vert in list_1:
        if not vert.both:
            del vert
    return None


def get_graph(points):
    # Kan sikkert optimalisere til å bare bruke "filter_points" metoden
    # flere ganger på rad.
    mst = make_mst(points)
    verticies = make_vertecies_with_neighbors(mst)
    # filter sjekk
    new_points, cntr = filter_points(verticies, points, count=True)
    # filter
    print('starter_for ' + str(cntr))
    for x in range(cntr):
        points, cntr = filter_points(verticies, points, count=False)
        print('Ferdig filter points')
        mst = make_mst(points)
        print('Ferdig mst')
        verticies = make_vertecies_with_neighbors(mst)
        # filter again
        # new_points = filter_points(new_verticies, new_points)
        # new_mst_ = make_mst(new_points)
        print('Ferdig med for')
    print('Ferdig med hele for')
    return verticies, mst, points


def get_graph_no_filter(points):
    # Kan sikkert optimalisere til å bare bruke "filter_points" metoden
    # flere ganger på rad.
    mst = make_mst(points)
    verticies = make_vertecies_with_neighbors(mst)
    # filter
    # points, cntr = filter_points(verticies, points, count=False)
    mst = make_mst(points)
    verticies = make_vertecies_with_neighbors(mst)
    return verticies, mst, points


def filter_no_times(points):
    # Kan sikkert optimalisere til å bare bruke "filter_points" metoden
    # flere ganger på rad.
    mst = make_mst(points)
    verticies = make_vertecies_with_neighbors(mst)
    # filter sjekk
    new_points, cntr = filter_points(verticies, points, count=True)
    # filter
    return verticies, mst, points


def left_right_master_vertex(verticies):
    # Sjekker hvilken av de med fire naboer som er til venstre
    # Setter den som er til venstre til B, høyre til C
    compare_masters = []
    for vertex in verticies:
        if len(vertex.neighbors) == 4:
            # har fire naboer er mastert punkt
            compare_masters.append(vertex)
    if compare_masters[0].x < compare_masters[1].x:
        compare_masters[0].category = 'B'
        compare_masters[1].category = 'C'
    else:
        compare_masters[1].category = 'B'
        compare_masters[0].category = 'C'
    return verticies


def find_point_by_category(verticies, category):
    # tar inn en liste med vertexes og returnerer den
    # som er aktuell.
    p1 = None
    for vertex in verticies:
        if vertex.category == category:
            p1 = vertex
            break
    return p1


def find_all_threes_verticies(verticies):
    # tar inn alle punktenen og returnerer dem som er treere
    verticies_3 = []
    for vertex in verticies:
        if vertex.nr_of_neighbors > 2:
            verticies_3.append(vertex)
    return verticies_3


def find_all_ones_verticies(verticies):
    v1 = []
    for vertex in verticies:
        if vertex.nr_of_neighbors == 1:
            v1.append(vertex)
    return v1


def find_a_b_c_verticies(verticies):
    # Finner A B og C punktene ved å ta utgangspunkt i B
    # punktet. Returenere punktene.
    # vil finne venstre punkt
    p1 = find_point_by_category(verticies, 'B')
    out1 = p1
    out2 = None
    out3 = None
    # vil bare bruke de med 3 eller flere naboer
    v_3 = find_all_threes_verticies(verticies)
    # har nå venstre hovedpunkt
    # vil sammenligne alle punktene i "left branches" med dette punktet
    # retning er ikke viktig
    for p2 in v_3:
        # sjekker alle grener som har punktet i seg
        # og punktet brukes til å sjekke vinkel mot.
        # if (p1 in branch.points) and (branch.category == ''):
        # Sjekker bare dem som har
        # utgangspunktet i seg.
        if p2 != p1:
            angle = calc_angle(p1, p2)
            if 160 < angle < 200:
                # her burde og endepunktet settes til en kategori.
                # lager en metode som setter kategori
                p2.category = 'A'
                out2 = p2
            elif (-20 < angle < 20) or (340 < angle < 20):
                p2.category = 'C'
                out3 = p2
    return out1, out2, out3


def find_points_above(verticies, category):
    # tar inne en kategori til et punkt (A, B, C)
    # returnerer en liste med alle punktene som er over,
    # sortert etter y verdi (første har lavest y)
    p1 = find_point_by_category(verticies, category)
    v_3 = find_all_threes_verticies(verticies)
    # v_1 = find_all_ones_verticies(verticies)
    # v_key = v_3 + v_1
    pnts_above = []
    if p1 is not None:
        for p2 in v_3:
            if p2 != p1:
                angle = calc_angle(p1, p2)
                if 80 < angle < 103:
                    pnts_above.append(p2)
                if (-100 < angle < -80) or (250 < angle < 290):
                    p2.category = 'root'
    pnts_above.sort(key=lambda vtx: vtx.y, reverse=False)
    return pnts_above


def categorize_horizontal(verticies, category):
    find_a_b_c_verticies(verticies)
    cnt = 0
    va = find_points_above(verticies, category)
    for vertex in va:
        cnt += 1
        vertex.category = category + 'T' + f'{cnt}'


def get_keypoints(verticies, points):
    for vertex in verticies:
        vertex.add_coords(points[vertex.idx, :])
    # finn firepunkt vertexenen
    left_right_master_vertex(verticies)
    categorize_horizontal(verticies, 'A')
    categorize_horizontal(verticies, 'B')
    categorize_horizontal(verticies, 'C')
    v3 = find_all_threes_verticies(verticies)
    v3.sort(key=lambda vtx: vtx.y, reverse=False)
    cnt = 0
    for vertex in v3:
        if vertex.category == '':
            cnt += 1
            vertex.category = 'D' + 'T' f'{cnt}'


def categorize_branches(branches, verticies):
    # Tar inn grener og kategoriserer dem i
    # forhold til hvilke kategor endepunktene har.
    v1 = find_all_ones_verticies(verticies)
    v1.sort(key=lambda vtx: vtx.x, reverse=False)
    dcat = []
    for vertex in v1:
        voe = vertex.other_end()  # punktet i andre ende, for kategori
        next_cat = voe.category
        if vertex.category == '':
            # Kan prøve å resette counteren dersom kategorien til det forrige punktet er annerledes.
            dcat.append(next_cat)  # kodesnutt for å resette teller, ved gren base skifte
            if len(dcat) > 1:
                res = dcat.count(next_cat)
                if res > 0:
                    cnt = res
                else:
                    cnt = 1
            else:
                cnt = 1
            vertex.category = next_cat + f'{cnt}'
    for branch in branches:
        branch.categorize()


# Kan kanksje lage en metode som tar inn et punkt, og finner alle treer
# punkter over det punktet, og kategoriserer dem

    # branch.category = cat
    # branch.set_category(p1, cat)
    # Må kanskje gjøre "set category" til slutt

#
# def categorize_branches_2(branches, vertices):
#     # test
#     categorize_branches(branches, vertices, 'B')
#     pass

if __name__ == '__main__':
    # Lager en punktliste
    xer = [3, 4, 5, 6, 6, 6, 6, 6, 7, 8, 9, 9, 9, 9, 9, 9, 9, 10, 11, 12, 13, 13, 13, 13, 14, 6, 6, 9.5]
    yer = [5, 5, 5, 5, 6, 7, 8, 9, 5, 5, 2, 3, 4, 5, 6, 7, 8, 5, 5, 5, 5, 6, 7, 8, 6, 4, 3, 5.5]
    points_ = np.stack((xer, yer), axis=1)
    mst_ = make_mst(points_)
    # vertices_, new_mst_, new_points_ = filter_multiple_times(points_)

    # blength = branches_[0].get_length()  # testing av lengde
    # categorize_branches(branches_, vertices_)
    # categorize_threes_verticies(vertices_, new_points_)

    # make branches
    # branches_ = make_branches(vertices_)
    # categorize_branches(branches_, vertices_)

    # kategorisere grener etter hvilken gren de hører til,
    # ved først å ta utgangspunkt i fire splittene, den som har
    # lavest x verdi er den til venstre

    # Løkke bare for å dobbeltsjekke grenene
    # for branche in branches_:
    #     print(branche)
    #     for point_ in branche.points:
    #         print(point_)

    # plt.subplot(121)
    # plot_mst(mst_, points_, 'test')
    # plt.subplot(122)
    # plot_mst(new_mst_, new_points_, 'test')
    # plt.show()
    #
