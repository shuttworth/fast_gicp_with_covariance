
* Ref: https://github.com/SMRT-AIST/fast_gicp

- Add some useful functions for both cpp & python
- Modify gicp as it utilizes raw covariance by following normalized_ellipse mode (not plane mode), in order to meet the scales for multiple 3D pointclouds
=> scale = scale / scale[1] .max(1e-3)
  
* note that cov = R*S*(R*S)^T = R*SS*R^T,   S = scale.asDiagonal();
* here, R = quaternion.toRotation();
* q = (q_x, q_y, q_z, q_w)
* R and SS can be obtained by SVD; R=U, scale**2 = singular_values.array()

Install for python
```shell
mkdir build
cd build
cmake ..
make -j8
cd ..
python setup.py install --user
```

python usage (see src/fast_gicp/python):

```python
import pygicp

gicp = pygicp.FastGICP()

gicp.set_input_target(target)
gicp.set_input_source(source)

# set covariance from quaternion and scale by following normalized_elipse
nparray_of_quaternions = nparray_of_quaternions_Nx4.flatten()
nparray_of_scales = nparray_of_scales_NX3.flatten()
gicp.set_target_covariance_fromqs(nparray_of_quaternions, nparray_of_scales) => 0.002180 sec
gicp.set_source_covariance_fromqs(nparray_of_quaternions, nparray_of_scales)

# compute covariance by following normalized_elipse
calculate_target_covariance() # compute covariance from given input target pointcloud
calculate_source_covariance() # compute covariance from given input source pointcloud

# after gicp.align()
correspondences, sq_distances = gicp.get_source_correspondence()
covariances = get_target_covariances()
covariances = get_source_covariances()
nparray_of_quaternions = get_target_rotationsq() => 0.00002277 sec
nparray_of_quaternions = get_source_rotationsq() 
nparray_of_scales = get_target_scales()          => 0.00002739 sec
nparray_of_scales = get_source_scales()
nparray_of_quaternions_Nx4 = np.reshape(nparray_of_quaternions, (-1,4))
nparray_of_scales_NX3 = np.reshape(nparray_of_scales, (-1,3))

```