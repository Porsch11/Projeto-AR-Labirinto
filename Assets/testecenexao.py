
import numpy as np
import socket
import numpy as np
import time
import random as rand
from random import shuffle, randrange
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

def chama_matrizes():
    print('\nmaux =\n ', maux)
    print('\nlab =\n ', lab)
    print('\nmaux_saida =\n ',maux_saida)
    print('\nlabirintos =\n ', labirintos )
    return labirintos


def cria_matriz():

    altura = int(input("digite o numero de linhas do labirinto: "))
    largura = int(input("Digite o número de colunas do labirinto: "))


    print('altura: ', altura)
    print('largura: ', largura)


    def make_maze(w=int(largura/2), h=int(altura/2)):
        vis = [[0] * w + [1] for _ in range(h)] + [[1] * (w + 1)]
        ver = [["01"] * w + ['0'] for _ in range(h)] + [[]]
        hor = [["00"] * w + ['0'] for _ in range(h + 1)]
        print(hor)

        def walk(x, y):
            vis[y][x] = 1

            d = [(x - 1, y), (x, y + 1), (x + 1, y), (x, y - 1)]
            shuffle(d)
            for (xx, yy) in d:
                if vis[yy][xx]:
                    continue
                if xx == x:
                    hor[max(y, yy)][x] = "01"
                if yy == y:
                    ver[y][max(x, xx)] = "11"
                walk(xx, yy)

        walk(randrange(w), randrange(h))
        for (a, b) in zip(hor, ver):
            maze = ''.join(a + ['\n'] + b)
            print(maze)
            
        global altura_ideal, largura_ideal

        global labirinto
        labirinto = []
        
        global maux, maux_saida
        maux = []
        maux_saida = []
        
        global posicao_final_linha, posicao_final_coluna
        posicao_final_linha = []
        posicao_final_coluna = []

        print('aqui: ', (len(hor[1][:]))-1)
        i = 0
        for i in range((len(hor))-1):

            count = int(largura//2)
            print('aqui2x:   ', count)
            posicao_separada_hor = []
            posicao_separada_ver = []
            e = 0
            for e in range(count):
                lista_hor = list(hor[i][e])
                print('lsita hor', lista_hor)
                separa_posicao_hor = []
                separa_posicao_hor = separa_posicao_hor + lista_hor
                posicao_separada_hor = (posicao_separada_hor + separa_posicao_hor)
                print('hor: ', posicao_separada_hor, e)

                lista_ver = list(ver[i][e])
                print('lista ver', lista_ver)
                separa_posicao_ver = []
                separa_posicao_ver = separa_posicao_ver + lista_ver
                posicao_separada_ver = (posicao_separada_ver + separa_posicao_ver)
                print('ver: ', posicao_separada_ver, e)

                e = e + 1
            posicao_separada_hor.append('0')
            posicao_separada_ver.append('0')

            borda_forcada = ['0']*largura
            teste = []
            type(np.array((teste)))
            teste.append(posicao_separada_hor)
            teste.append(posicao_separada_ver)
            labirinto = labirinto + teste
            print('i  ', i)
            i = i+1

        if largura % 2 == 0:
            borda_forcada = ['0']*(largura+1)
            posicao_final_linha = 1
            posicao_final_coluna = largura - 1
            largura_ideal = largura + 1
            if altura % 2 == 0:
                altura_ideal = altura + 1
            else:
                altura_ideal = altura
            
            
        else:
            largura_ideal = largura
            borda_forcada = ['0']*largura
            posicao_final_linha = 1
            posicao_final_coluna = largura - 2
            if altura % 2 == 0:
                altura_ideal = altura + 1
            else:
                altura_ideal = altura
            
        maux = np.zeros(((altura_ideal), (largura_ideal)), dtype=np.float64)
        maux_saida = np.zeros(((altura_ideal), (largura_ideal)), dtype=np.float64)
            

        labirinto.append(borda_forcada)
        print("********")
        print('labirinto: {}', labirinto)
        print("********")
        print('linhas: ', (len(labirinto[:][0])))
        print('colunas: ', (len(labirinto[0][:])))
        # print(i)
        # print(abacaxi)
        print('vis: ',vis)

    make_maze()

    
    print('')
    print(posicao_final_linha)
    print(posicao_final_coluna)
    global labirintos
    labirintos = np.array((labirinto), dtype=np.float64)
    posicao_linha = 1
    posicao_coluna = 1
    maux[posicao_linha][posicao_coluna] = 1
    maux_saida[posicao_final_linha][posicao_final_coluna] = 2

    print(maux)
    print('')
    print(labirintos[:][:])
    global lab
    lab = labirintos + maux + maux_saida
    print('')
    print(lab)

    #esta função cria blocos livre no local de parades para possibilitar mais de um caminho até saida, são gerados aleatóriamente
    #(não garante um segundo caminho)

    def caminho_livre():
        altura_teste = randrange(1,altura-1)
        largura_teste = randrange(1,largura-1)
      

        if lab[altura_teste][largura_teste] == 0:
            lab[altura_teste][largura_teste] = 1
            print('altura ', altura_teste)
            print('largura ', largura_teste)
            print('Encontrou caminho livre', lab)
        else:
            print('Não encontrou caminho livre')
            caminho_livre()
        

    if (altura*largura) < 100:
        n  = 1
        for i in range(n):
            caminho_livre()
            i = i+1

        
    elif 300 > (altura*largura) >= 100:
        n  = 2
        for i in range(n):
            caminho_livre()
            i = i+1
        
    elif 500 > (altura*largura) >=300:
        n = 6
        for i in range(n):
            caminho_livre()
            i = i+1

    elif 2000 > (altura*largura) >=500:
        n = 5
        for i in range(n):
            caminho_livre()
            i = i+1

    else:
        print('caminho maior')

    global lab_puro
    lab_puro = lab - maux - maux_saida

    plt.figure(figsize=(10, 10))

    mapa_cores = {
        0: '#FF0000',  # vermelho
        1: '#FFFFFF',  # branco
        2: '#000000',  # preto
        3: '#0000ff',
    }

    cores = list(mapa_cores.values())

    cmap = LinearSegmentedColormap.from_list('', cores)
    plt.imshow(lab, cmap=cmap)
    cbar = plt.colorbar()
    cbar.remove()

    plt.tick_params(left=False, right=False, labelleft=False,
                    labelbottom=False, bottom=False)
    plt.show()
    
    
def main():
    
    cria_matriz() 
    like = (input('\nVocê quer utilziar esta matrix? S/N: \n'))
    like = like.upper()
    print()

    if like == 'S':
        chama_matrizes()
    elif like == 'N':
        main()
    else:
        print("Erro de comando")
        

'''      
def encontra_robo(lab):
    local = np.where(lab == 2)
    print('\nlocal: \n',local)
    return local
    '''

main()

def comunicacao():

    host, port = "127.0.0.1", 25001
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((host, port))

    matriz = []

    x = []

    labirintos[1][len(labirintos[1][:])-1] = 1
    matriz = labirintos.astype("int")
    '''matriz = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0],
        [0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0],
        [0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0],
        [0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0],
        [0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])'''

    x = [len(matriz[:][0]),len(matriz[0][:])]
    print(x)

    time.sleep(0.5)  # sleep 0.5 sec
    # Converting Vector3 to a string, example "0,0,0"
    posString = ','.join(map(str, matriz))
    print(posString)

    xstr = ','.join(map(str, x))
    print(xstr)


    sock.sendall(xstr.encode("UTF-8"))
    # Converting string to Byte, and sending it to C#
    sock.sendall(posString.encode("UTF-8"))
    # receiveing data in Byte fron C#, and converting it to String
    receivedData = sock.recv(1024).decode("UTF-8")
    
def construa_estados(N):
    estados = np.zeros((altura_ideal, largura_ideal), dtype=np.float64)
    for i in range(N):
        for j in range(N):
            estados[i,j] = ((i*N)+j)

    estados = estados
    return estados
    
def construa_direcoes(N,princ,Q):
    n_estados = range(N*N-1)
    estados = construa_estados(N)
    acoes = [-N, +N, +1, -1]
    vetor_direcao=['c','b','d','e']
    direcao = np.zeros((altura_ideal, largura_ideal), dtype=str)

    for i in range(N):
        for j in range(N):
            posicao = np.where(Q[(i*N)+j][:]==max(Q[(i*N)+j][:]))
            direcao[i][j] = vetor_direcao[posicao[0][0]]

    for i in range(N):
        for j in range(N):
            if(princ[i][j]==0):
                direcao[i][j]='o'
    return direcao

def encontra_sentido(sentidos,sentido_atual): #retorna o sentido do robo  em h, que é o index na lista de sentidos
    h = sentidos.index(sentido_atual) 
    return h
    
def ler_sensores(labirinto, sentido_atual, Q, estado_atual_aux,maux):
  
    estado_atual = int(estado_atual_aux)
    leituras = np.zeros(3, dtype=np.float64) #retornar um array de 0 (1X3)
    sentidos = ['c','b','d','e'];  #sentidos possíveis

    M_sentidos = [["e","c","d"], #c     #define o novo sentido com base no sentido atual
                    ["d","b","e"], #b     # se x = 2 y = 1 o robo estará com o sentido direita e se locomoverá para a esquerda, seu novo sentido será cima
                    ["c","d","b"], #d     # pois a esquerda do robo que está direcionado para a direita é o nosso referencial cima.
                    ["b","e","c"]] #e

    # a cada duas colunas uma direção  (primeira coluna representa o deslocamento na linha e a segunda o deslocamento na coluna)
                        #Direções   #E      #F      #D     #Sentidos
    sensores_direcao = np.array([[ 0, -1, -1,  0,  0, +1], #c (usa esta linha para ler em volta do robo caso o sentido atual seja c)
                                    [ 0, +1, +1,  0,  0, -1], #b (usa esta linha para ler em volta do robo caso o sentido atual seja b)
                                    [-1,  0,  0, +1, +1,  0], #d (usa esta linha para ler em volta do robo caso o sentido atual seja d)
                                    [+1,  0,  0, -1, -1,  0]]) #e(usa esta linha para ler em volta do robo caso o sentido atual seja e)


    h = encontra_sentido(sentidos,sentido_atual)
    constantes = sensores_direcao[h,:]
    
    local = localiza_robo(maux)
    #print(local)
    #print(local, labirinto, maux)
    indice_sentidos = []
    index = 0  

    for i in leituras:  #Preenche no array leitura o ambiente a sua volta, na posição [index]  o local atual + a direção a seguir (sensores_direção) a linha utilizada é selecionada por h(encontra_sentido)
        
        #DFE[index]=[local[0]+constantes[index*2], local[1]+constantes[index*2+1]
        print(local)
        leituras[index] = labirinto[local[0]+constantes[index*2], local[1]+constantes[index*2+1]]
        hq = encontra_sentido(sentidos,M_sentidos[h][index])
    #print(hq)
        #print(estado_atual)
        #print(leituras)
        if(leituras[index] == 0):
            Q[estado_atual][hq] = -np.inf
            #Nave(esquerda(1),esquerda(2)) = 0;
            #R(1)=-inf;
        else:
            indice_sentidos.append(hq)
    # R(1)=-1;
    # Nave(esquerda(1),esquerda(2)) = 1;
        if(Q[estado_atual,hq] == -np.inf):
            Q[estado_atual,hq] = -1

        index = index + 1

    if(sum(leituras)==0):

        if(sentido_atual =='c'):
            aux="b"
        
        elif(sentido_atual =='b'):
            aux="c"
        elif(sentido_atual =='d'):
            aux="e" 
        elif(sentido_atual =='e'):
            aux="d"

        #print(aux)
        h = encontra_sentido(sentidos,aux);
        hq = encontra_sentido(sentidos,M_sentidos[h][1])
        indice_sentidos.append(hq) 
        
    return leituras, Q, indice_sentidos  

def anda_robo (maux, sentido_atual, direcao_futura):

    direcoes = ["F", "E", "D"]
    sentido = ["c", "b", "d", "e"]

    x = encontra_sentido(sentido, sentido_atual) # retorna um valor para a linha entre 0 e 3 para os sentidos
    y = encontra_direcao_futura(direcoes, direcao_futura) # retorna um valor para a coluna entre 0 e 2 para as direções
    
    
            #F #E #D     Define o movimento na linha atual do robo
    a = [[-1, 0, 0],   #c (usa esta linha para mover o robo caso o sentido atual seja c)
        [ 1, 0, 0],   #b (usa esta linha para mover o robo caso o sentido atual seja b)
        [ 0,-1, 1],   #d (usa esta linha para mover o robo caso o sentido atual seja d)
        [ 0, 1,-1]]   #e (usa esta linha para mover o robo caso o sentido atual seja e)

            #F #E #D     Define o movimento na coluna atual do robo
    b = [[0, -1, +1],  #c (usa esta linha para mover o robo caso o sentido atual seja c)
        [ 0, +1,-1],  #b (usa esta linha para mover o robo caso o sentido atual seja b)
        [ 1,  0, 0],  #d (usa esta linha para mover o robo caso o sentido atual seja d)
        [-1,  0, 0]]  #e (usa esta linha para mover o robo caso o sentido atual seja e)

    local = localiza_robo(maux) 

    maux[local[0][0]+a[x][y]][local[1][0]+b[x][y]]=1  #adiciona 1 no novo local em que o robo está
    maux[local[0][0]][local[1][0]] = 0 #adiciona zero no local antigo em que o robo estava no labirinto auxiliar

    #%// sentido_atual=M_sentidos[x][y];
    sentido_atual = proximo_sentido(sentido_atual, direcao_futura);

    return maux, sentido_atual

def get_index_maxQ(indice_sentidos,Q_atual):
    #print(indice_sentidos)
    #print(Q_atual)
    temp_indices_aux = np.where(Q_atual==max(Q_atual[indice_sentidos]))
    temp_indices = []
    #print(temp_indices_aux)

    index = 0
    for i in range(len(temp_indices_aux[0])):
        temp_indices.append(temp_indices_aux[0][index])
        #print(temp_indices_aux[0][index])
        index = index+1

    indice = np.random.choice(temp_indices, size=1)
    indice = indice[0]
    
    
    return indice

def encontra_direcao_futura(direcao, direcao_futura):
    h = direcao.index(direcao_futura)
    return h

def encontra_robo(labirinto): 
    local = np.where(labirinto == 2)
    return local

def localiza_robo(maux):
  local = np.where(maux== 1)
  return local

def proximo_sentido(sentido_atual, direcao_futura):

    direcoes = ["F", "E", "D"]
    sentido = ["c", "b", "d", "e"]

    x = encontra_sentido(sentido, sentido_atual);     # sentido atual  (retorna um valor para a linha entre 0 e 3 para os sentidos)
    y = encontra_direcao_futura(direcoes, direcao_futura);  # direcao futura (retorna um valor para a coluna entre 0 e 2 para as direções)
    
                   #F  #E  #D
    M_sentidos = [["c","e","d"], #c     #define o novo sentido com base no sentido atual
                  ["b","d","e"], #b     # se x = 2 y = 1 o robo estará com o sentido direita e se locomoverá para a esquerda, seu novo sentido será cima
                  ["d","c","b"], #d     # pois a esquerda do robo que está direcionado para a direita é o nosso referencial cima.
                  ["e","b","c"]] #e


    sentido_atual = M_sentidos[x][y]
    return sentido_atual

princ = lab_puro

l1 = 1-1
l2= len(princ[0][:])-1 
l3 = len(princ[:][0])-1 

n_iteracoes = 30; #numero de tentativas (épocas)
gamma = 0.8;
learning_rate = 0.4;
actions = 4;  #Cima, Esquerda, Direita, Baixo [c,e,d,b]
exprate=0.7;
exploration_rate = exprate;
N = len(princ)

largura_labirinto = len(princ[0][:])#array de listas - pega o tamanho da lista no index zero
altura_labirinto = N #array de listas - pega o primeiro elemento de todas as listas

sentidos = ['c','b','d','e']

sentido_inicial = "e" #sentido da posição inicial
posicao_inicial = 1 #linha da posicão inicial do agente
posicao_ini_coluna = 1  #coluna da posição inicial do agente

posicao_fin_linha = 1
posicao_fin_coluna = len(labirintos) - 1

posicao_final = [posicao_fin_linha,posicao_fin_coluna]

estados = construa_estados(N)

maux = np.zeros((largura_labirinto,altura_labirinto), dtype=np.float64) 
#labirinto = np.zeros((largura_labirinto, altura_labirinto), dtype=np.float64)
#Nave = np.ones((N,N),dtype=np.float64)*-1;
Q = np.zeros((N*N,4),dtype=np.float64)
maux[posicao_inicial][posicao_ini_coluna] = 1 #adiciona o robo(agente) no labirinto auxiliar
estado_inicial = estados[posicao_inicial,posicao_ini_coluna]
estado_final = estados[posicao_fin_linha,posicao_fin_coluna]

aux_directions = construa_direcoes(N,princ,Q)
aux = sentido_inicial
matriz_acao = [-N, +N, +1, -1]

i = 1
recompensa = 0
impressao = 1

labirinto = princ + maux #adiciona o robo(agente) no labirinto(ambiente)
      #no labirinto o robo(agente) será localizado pelo número 2
labirinto

print(labirinto)
i = 0
epoca = 0
for epoca in range(n_iteracoes): # inicia com n = 500 
  #inicia com estético
  
  maux = np.zeros((largura_labirinto,altura_labirinto), dtype=np.float64) 
  maux[posicao_inicial][posicao_ini_coluna] = 1
  sentido_atual = sentido_inicial
  estado_atual = estado_inicial
  aux = sentido_atual #inicia com aux = 1
  #labirinto = maux + princ
  #print(labirinto,maux)
  #princ[posicao_inicial][posicao_ini_coluna] = 0
  
  k_index = 0 
  vetor = [ ]
  vetor.append(aux) # vetor = []   
  trajetoria = []
  trajetoria_epoca = []

  n=0
  #print(sentido_atual,Q, estado_atual)
  while(estado_atual != estado_final):
    #print('epoca é {}'.format(epoca))
    n=n+1
    labirinto = maux + princ
    leituras, Q, indice_sentidos = ler_sensores(labirinto, sentido_atual, Q, estado_atual,maux) #inicia com [(0,1,0)]
    #print(labirinto)
    labirinto = maux + princ
    
    #print(Q[int(estado_atual), :]),
    
    if((epoca%50 == 0) and (epoca >= 10)):
        
        plt.figure(figsize=(5, 5))

        mapa_cores = {
            0: '#FF0000',  # vermelho
            1 : '#FFFFFF',  # branco
            2 : '#000000',  # preto
        }

        cores = list(mapa_cores.values())

        cmap = LinearSegmentedColormap.from_list('', cores)
        plt.imshow(labirinto, cmap=cmap)
        cbar = plt.colorbar()
        cbar.remove()
        grid = plt.grid(False)
        
        plt.tick_params(left = False, right = False , labelleft = False ,
                        labelbottom = False, bottom = False)
        plt.figure(figsize=(5, 5))
        plt.show()
        
        #print(labirinto)
        
 
    randomico = rand.random()    
    if(randomico < exploration_rate):
      acao_escolhida = np.random.choice(indice_sentidos, size=1)
      acao_escolhida = acao_escolhida[0]
     #print(acao_escolhida)
    else:
      indice = get_index_maxQ(indice_sentidos,Q[int(estado_atual)][:])
      #print(indice)
      acao_escolhida = indice
    #print(acao_escolhida)
    sentido_atual = sentidos[acao_escolhida]
    print(sentido_atual)
    print(acao_escolhida)
    maze_exp_index = estado_atual + matriz_acao[acao_escolhida]
    
    #print(maze_exp_index)
    trajetoria.append(maze_exp_index)
    trajetoria_epoca.append(maze_exp_index)
    k_index=k_index+1
    i = i + 1

    lc = np.where(estados==maze_exp_index)

        #if(maze[lc[0][0],lc[1][0]] == 0):
	       # reward = -inf; % OBS: O Robo não chega a bater na parede, a recompensa -inf é dada no m-file ler_Sensores.m
       # else
           # reward = -100; % A cada passo do robô, ele recebe -100 como recompensa
    reward = -100  
    if(maze_exp_index==estado_final):
           reward = 1000
       
    Q[int(estado_atual), int(acao_escolhida)] = reward + gamma*max(Q[int(maze_exp_index),:])

    estado_atual = maze_exp_index
    maux = np.zeros((largura_labirinto,altura_labirinto), dtype=np.float64)
    maux[lc[0][0],lc[1][0]] = 1
    labirinto = princ + maux

  maux[posicao_inicial][posicao_ini_coluna] = 1
  directions = construa_direcoes(N,princ,Q)
    
  if((aux_directions == directions).all()):
    exploration_rate = 0.0
  else:
    exploration_rate = exprate
    
  aux_directions = directions 
  epoca = epoca + 1

  if((impressao != 0) and (epoca% 25 ==0)):
    print("Final da  epoca {}\n".format(epoca))
    print(directions)
    
    k = []
    o = []
    for i in range(len(trajetoria_epoca)):
      ko = np.where(estados == trajetoria_epoca[i])
      #trajetoria_epoca[i] = linhacoluna[0]
      if trajetoria_epoca[i] in estados:
        k.append(ko[0][0])
        o.append(ko[1][0])

        maux[k,o]=1
        caminho_escolhido = maux
    labirinto = maux + princ 
    caminho_escolhido = maux
    print('caminho:',caminho_escolhido)
    print('lab',labirinto)
    print('\ndirections: \n',directions)

direcoes = construa_direcoes(N,labirinto,Q)
print('direçoes: ',direcoes)
print('caminho_escolhido: ',caminho_escolhido)

comunicacao()