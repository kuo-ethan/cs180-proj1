TODO
main: given a 3 panel image, output a colored image
1. Split up into 3 images
2. Implement utility functions for euclidean and NCC metrics
- (arr1, arr2) -> number
3. Implement align function 
- (arr, base, metric, displacement_factor=15) -> displacement vector
4. Compute aligned green and aligned red images using displacement vectors
5. Construct the colored image by stacking