def change_names(tree, names_dict):
    """
    Changes names on the tree, returns re_order for the matrix
    """
    re_order = []
    for clade in tree.find_clades():
        try:
            clade_n = str(clade.name)
            clade.name = str(names_dict[clade_n])
            re_order.append(clade_n)
        except KeyError:
            pass
    return tree, re_order

def getNewick(node, newick, parentdist, leaf_names):
    if node.is_leaf():
        return "%s:%.2f%s" % (leaf_names[node.id], parentdist - node.dist, newick)
    else:
        if len(newick) > 0:
            newick = "):%.2f%s" % (parentdist - node.dist, newick)
        else:
            newick = ");"
        newick = getNewick(node.get_left(), newick, node.dist, leaf_names)
        newick = getNewick(node.get_right(), ",%s" % (newick), node.dist, leaf_names)
        newick = "(%s" % (newick)
        return newick

def get_color_f(condition_D, condition_healthy, condition_normal):
    def colors(label):
        n = str(label)
        if n in condition_D.values():
            return 'r'
        elif n in condition_healthy.values():
            return 'g'
        elif n in condition_normal.values():
            return 'b'
        else:
            return (0,0,0)
    return colors