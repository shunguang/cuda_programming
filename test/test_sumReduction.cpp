#include "03_sum_reduction/include/sumReduction.cuh"
using namespace std;

int test_sumReduction(int argc, char** argv)
{
	int ret;
	int k = 0;
	int n0 = 1 << 13;
	while ( k++ < 200) {
		int N = 0;
		for (int i = 0; i < 5; i++) {
			N = n0*(1 << i);
			printf("i=%d, N=%d\n", i, N);
			for (int j = 0; j < 1; j++) {
				//ret = test_sumReduction1a(N);
				//ret = test_sumReduction2a(N);
				ret = test_sumReduction6(N);  //need lots of shared memoery
				if (ret < 0) { break; }
			}
			if (ret < 0) { break; }
		}

		for (int i = 0; i < 1; i++) {
			N = n0*(1 << i) + rand();
			printf("i=%d, N=%d\n", i, N);
			for (int j = 0; j < 1; j++) {
				ret = test_sumReduction1b(N);
				ret = test_sumReduction2b(N);
				//ret = test_sumReduction6(N);  //need lots of shared memoery
				if (ret < 0) { break; }
			}
			if (ret < 0) { break; }
		}

		if (ret < 0) { break; }
	}
	return 0;
}


