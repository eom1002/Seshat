#include "wtime.h"
#include <bits/stdc++.h>
using namespace std;
const int mx = 1e5;
int a[mx+4];
// a[0] == 0, prime number
// a[1] == 1, not prime
void sieve() {
    a[0] = a[1] = 1;
//#pragma omp parallel for
    for (int i = 2; i*i <= mx; i++) {
        if (!a[i]) {
//#pragma omp parallel for 
            for (int j = i<<1; j <= mx; j+=i) a[j] = 1;
        }
    }
}

int main() {
    int i=700;
    double t_beg=Wtime();
//#pragma omp parallel for
    for(int n=0;n<i; n++)  {
     sieve();
     scanf("%d", &n);
     if (a[n] == 0) printf("%d is a Prime number\n", n);
     else printf("%d is Not a Prime number\n", n);
     }
    double t_end=Wtime();
    std::cout << "finding all the primes took " << t_end - t_beg << "s\n";
    return 0;
}
