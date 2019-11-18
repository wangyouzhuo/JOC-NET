import h5py


class grid_map():

    def __init__(self,first_line,h5_file_path):
        self.grids = [ [-1] * 40 for i in range(40)]
        self.first_line = first_line
        for i in range(len(self.first_line)):
            self.grids[29][10+i] = first_line[i]
        self.visited_id_set = set()
        for item in first_line:
            self.visited_id_set.add(item)
        self.h5_file = h5py.File(h5_file_path, 'r')
        self.graph = self.h5_file['graph'][()]
        print(self.graph[8])


    def visit(self):
        for i in range(29,-1,-1):
            for item in self.grids[i]: # item 是 节点id
                if item >= 0:
                    neighbour_nodes = self.graph[item]
                    # print("neighbour_nodes:",neighbour_nodes,'node',item)
                    for node_id in neighbour_nodes:
                        if node_id not in self.visited_id_set and node_id>0:
                            column_id = self.grids[i].index(item)
                            row_id = i
                            self.grids[row_id-1][column_id] = node_id
                            self.visited_id_set.add(node_id)
            print(i-1,self.grids[i-1])






if __name__ == '__main__':

    first_line = [0,8,16,24,32,40,44,88,148,224,292,324]
    file_path = '/home/wyz/PycharmProjects/distill-based-multi-target-visual-naviigation/data/'
    scene_name = 'bedroom_04.h5'
    path = file_path+scene_name
    g_map = grid_map(first_line,path)
    g_map.visit()








