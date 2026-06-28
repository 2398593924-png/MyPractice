#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <conio.h>
#include <windows.h>
#include <time.h>

typedef struct {
    bool** data;
    bool** food;
    bool food_exist;
    int size;
    int Score;
} GameMap;

typedef struct Node {
    struct Node* next;
    int x;
    int y;
    int state;
} SNode, *Ptr_SNode;
// up 1, down 2, left 3, right 4

void Snake_State(Ptr_SNode& S, int x, int y, int state, Ptr_SNode next){
    if (!S) return;
    S->x = x;
    S->y = y;
    S->state = state;
    S->next = next;
}

void GameMap_Init(GameMap& GM, int map_size){
    GM.size = map_size;
    GM.data = (bool**)calloc(map_size, sizeof(bool*));
    GM.food = (bool**)calloc(map_size, sizeof(bool*));
    GM.food_exist = false;
    GM.Score = 0;
    if (!(GM.data) || !(GM.food)) return;
    for (int i = 0; i < map_size; i++){
        GM.data[i] = (bool*)calloc(map_size, sizeof(bool));
        GM.food[i] = (bool*)calloc(map_size, sizeof(bool));
    }
}

void GameMap_ShowFrame(GameMap GM){
    system("cls");
    for (int i = 0; i < GM.size + 2; i++){
        for (int j = 0; j < GM.size + 2; j++){
            if ((i == 0) || (i == GM.size + 1)){
                printf("-");
                continue;
            }
            if ((j == 0) || (j == GM.size + 1)){
                printf("|");
                continue;
            }

            if (GM.food[i - 1][j - 1]){
                printf("@");
                continue;
            }

            if (GM.data[i - 1][j - 1]){
                printf("#");
            }else{
                printf(" ");
            }

        }
        printf("\n");
    }
}

void Snake_Init(Ptr_SNode& S, GameMap GM){
    int half_xy = GM.size / 2;
    S = (Ptr_SNode)malloc(sizeof(SNode));
    Ptr_SNode S_n = (Ptr_SNode)malloc(sizeof(SNode));
    if (!S_n) return;
    Snake_State(S, half_xy, half_xy, 3, S_n);
    Snake_State(S->next, half_xy - 1, half_xy, 3, NULL);
}

void Snake_Add(Ptr_SNode& S){
    Ptr_SNode p = S;
    while(p->next){
        p = p->next;
    }
    Ptr_SNode S_n = (Ptr_SNode)malloc(sizeof(SNode));
    if (!S_n) return;
    p->next = S_n;
    switch(p->state){
        case 1:
            Snake_State(S_n, p->x, p->y - 1, p->state, NULL);
            break;
        case 2:
            Snake_State(S_n, p->x, p->y + 1, p->state, NULL);
            break;
        case 3:
            Snake_State(S_n, p->x - 1, p->y, p->state, NULL);
            break;
        case 4:
            Snake_State(S_n, p->x + 1, p->y, p->state, NULL);
            break;
    }
}

void Snake_Update(Ptr_SNode& S, GameMap& GM){
    if (!S) return;
    
    Ptr_SNode nodes[1000];
    int count = 0;
    Ptr_SNode p = S;
    while(p){
        nodes[count++] = p;
        p = p->next;
    }
    
    for(int i = count - 1; i > 0; i--){
        nodes[i]->x = nodes[i-1]->x;
        nodes[i]->y = nodes[i-1]->y;
        nodes[i]->state = nodes[i-1]->state;
    }
    
    switch(S->state){
        case 1:
            S->y--;
            break;
        case 2:
            S->y++;
            break;
        case 3:
            S->x--;
            break;
        case 4:
            S->x++;
            break;
    }
}

void GameMap_Update(GameMap& GM, Ptr_SNode S){
    Ptr_SNode p = S;
    for (int i = 0; i < GM.size; i++){
        for (int j = 0; j < GM.size; j++){
            GM.data[i][j] = false;
        }
    }

    while(p){
        GM.data[p->x][p->y] = true;
        p = p->next;
    }
}

void GameMap_GenerateFood(GameMap& GM, Ptr_SNode S){
    if (GM.food_exist) return;
    srand((unsigned int)time(NULL));
    Ptr_SNode p = S;
    bool flag = true;
    int fx, fy;
    while(flag){
        flag = false;
        fx = rand() % (GM.size - 2) + 1;
        fy = rand() % (GM.size - 2) + 1;
        while(p){
            if (p->x == fx && p->y == fy){
                flag = true;
                break;
            }
            p = p->next;
        }
    }
    GM.food[fx][fy] = true;
    GM.food_exist = true;
}

void Snake_Check(GameMap& GM, Ptr_SNode& S){
    Ptr_SNode p = S;
    while(p->next){p = p->next;}
    if (GM.food[S->x][S->y]){
        GM.food[S->x][S->y] = false;
        GM.food_exist = false;
        Snake_Add(S);
        GM.Score += 5;
    }

}

bool Game_Check(GameMap GM, Ptr_SNode S){
    if (S->x <= 0 || S->y <= 0 || S->x >= GM.size - 1 || S->y > GM.size - 2){
        return false;
    }
    Ptr_SNode p = S->next;
    while(p){
        if (S->x == p->x && S->y == p->y){
            return false;
        }
        p = p->next;
    }
    return true;
}

int main(){
    GameMap GM;
    Ptr_SNode S;
    GameMap_Init(GM, 20);
    Snake_Init(S, GM);
    int ch, ext;
    while(true){
        if (!Game_Check(GM, S)){
            printf("YOU LOSE!");
            system("pause");
            return 0;
        }
        Snake_Check(GM, S);
        Snake_Update(S, GM);
        GameMap_Update(GM, S);
        GameMap_GenerateFood(GM, S);
        GameMap_ShowFrame(GM);
        printf("\nSCORE = %d\n", GM.Score);
        //KeyBoard
        if (_kbhit()){
            ch = _getch();
            if (ch == 27){
                break;
            }
            if (ch == 224){
                ext = _getch();
                switch(ext){
                    case 72:
                        S->state = 3;
                        break;
                    case 80:
                        S->state = 4;
                        break;
                    case 75:
                        S->state = 1;
                        break;
                    case 77:
                        S->state = 2;
                        break;
                }
            }
        }
        //Keyboard_End
        printf("%d, %d\n", S->x, S->y);
        Sleep(200);
    }
    return 0;
}