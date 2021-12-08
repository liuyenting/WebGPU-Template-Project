# MP1: Vector Addition

## Objective
The purpose of this lab is for you to become familiar with using the CUDA API by implementing a simple vector addition kernel and its associated host code as shown in the lectures.

## Questions
1. **How many floating operations are being performed in the vector add kernel? Give your answer in terms of N and explain.**
    Assuming N represents the total number of elements in the vector, combine with how question 2 is asked, there should be N floating operations (add).

2. **How many global memory bytes are read and written by the vector add kernel? Give your answer in terms of N. Please give separate answers for the bytes read and written.**
