from tabulate import tabulate

def draw():
    data = [
            ["","Infer" , 2],
            ["","CodeQL" , 3],
            ["All static analysis tools", "",5],
            ["","KLEE ", 2],
            ["", "MoKLEE ", 4],
            ["Symbolic execution engines", "",6],
            ["Fuzzing tool", "AFL++ [29]", 8],
            ["","Vuldeepecker",1],
            ["","Funded ",2],
            ["","Devign ",1],
            ["","ReVeal ",2],
            ["","ReGVD ",2],
            ["","LineVul ",3],
            ["","LineVD ",3],
            ["","CodeXGLUE ",3],
            ["","GraphcodeBERT ",2],
            ["","ContraFlow ",3],
            ["DL based on static code information", "", 22],
            ["DL based on dynamic information","LIGER ",16],
            ["","Concoction",31]]

    print(tabulate(data, headers=["Categories", "Approaches", "#vuln"]))

draw()