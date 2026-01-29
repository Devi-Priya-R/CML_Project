#include <stdio.h>
#include <string.h>
int main() 
{
    char tree[100];
    int n, i;
    printf("Enter number of nodes: ");
    scanf("%d", &n);
    printf("Enter tree nodes :\n");
    for(i = 0; i < n; i++) 
        scanf(" %c", &tree[i]);
    void inorder(int index) 
    {
        if(index >= n || tree[index] == '#') return;
        inorder(2 * index + 1);
        printf("%c ", tree[index]);
        inorder(2 * index + 2);
    }
    void preorder(int index) 
    {
        if(index >= n || tree[index] == '#') return;
        printf("%c ", tree[index]);
        preorder(2 * index + 1);
        preorder(2 * index + 2);
    }
    void postorder(int index) 
    {
        if(index >= n || tree[index] == '#') return;
        postorder(2 * index + 1);
        postorder(2 * index + 2);
        printf("%c ", tree[index]);
    }
    printf("\nInorder: ");
    inorder(0);
    printf("\nPreorder: ");
    preorder(0);
    printf("\nPostorder: ");
    postorder(0);
    return 0;
}