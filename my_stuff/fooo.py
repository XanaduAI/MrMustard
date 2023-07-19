import numpy as np

from mrmustard.lab.representations.data.gaussian_data import GaussianData
from mrmustard.utils.misc_tools import general_factory

class XXX:

    def test_gaussian_resulting_from_multiplication_is_correct(self, TYPE, c, dim):
        X = np.random.rand(dim*2) # TODO: can this be moved into a parameterize fixture which would call dim?
        C = 42
        cov_input_a = np.eye(dim*2)# random_covariance(dim)
        mean_input_a = np.random.rand(dim*2)
        c_input_a = C
        a_params = {'cov': cov_input_a, 'means': mean_input_a, 'coeffs':c_input_a}
        input_gaussian_state_a = general_factory(TYPE, **a_params)

        cov_input_b = np.eye(dim*2)#random_covariance(dim)
        mean_input_b = np.random.rand(dim*2)
        c_input_b = C 
        b_params = {'cov': cov_input_b, 'means': mean_input_b, 'coeffs':c_input_b}
        input_gaussian_state_b = general_factory(TYPE, **b_params)

        output_gaussian_state = input_gaussian_state_a * input_gaussian_state_b
        cov_output = output_gaussian_state.cov
        mean_output = output_gaussian_state.means
        c_output = output_gaussian_state.c

        gaussian_of_input_a = self._helper_gaussian(cov_input_a, mean_input_a, c_input_a, X)
        gaussian_of_input_b = self._helper_gaussian(cov_input_b, mean_input_b, c_input_b, X)
        gaussian_of_output = self._helper_gaussian(cov_output, mean_output, c_output, X)

        assert isinstance(gaussian_of_input_a, np.ndarray)
        assert isinstance(gaussian_of_input_b, np.ndarray)
        assert isinstance(gaussian_of_output, np.ndarray)
        assert np.allclose(gaussian_of_input_a * gaussian_of_input_b, gaussian_of_output)
        
    def _helper_gaussian(self, covariance, mean, c, x) -> np.ndarray:
        precision_mat = np.linalg.inv(covariance)
        gaussian = c * -np.transpose(np.exp(x, mean)) * precision_mat * (x - mean)
        return np.asarray(gaussian)
    

x = XXX()
x.test_gaussian_resulting_from_multiplication_is_correct(GaussianData, 2, 3)
print("yo")