# Overview
This code provides the experiments done in 'Privacy-preserving Median Selection and Secure Aggregation in Federated Learning'.

# Experiments
## 1. Approximate median selection performance
In Median_Analysis.ipynb, we show approximate median is almost smae as exact median.
## 2. Approximate median aggregator performance in FL
In FL_ApproxMed.ipynb, we compare the performance of average aggregator, median aggregator, and approximate median aggregator in FL.
We implemented approximate median aggregator with chunk size 5, when the array length is 100.
In order to reproduce the experiment done in the paper, only have to do is run 'global_update' function in .ipynb file.
The inputs of function 'global_update' is global epoch, local epoch, model, aggregator, and attack case.
Attack case is byzantine failure case is FL, and for details, refer to the paper.
## 3. MPC performance
With client.py and server.py, we implement privacy-preserving median selection algorithm using Multi-party computation (MPC).
We implemented with python socket programming.
In order to reproduce the experiment, we have to run server.py first (to prepare socket), and client.py next.
Discriptions of functions are in server.py.
To be specific, we implemented following functions in MPC.
- MPC multiplication (mul_share)
- naive privacy-preserving comparison algorithm (sort1, pa_sort1)
- our privacy-preserving comparison algorithm (sort2, pa_sort2)
- privacy-preserving swapping algorithm (swap, pa_swap)
- Median selection algorithms (array length = 3, 4, 5, 32, 64, 128)
- Approximate median selection algorithms (array length 25, 32, 64, 125, 128, with chunk size 5)
