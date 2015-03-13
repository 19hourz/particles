#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/random.h>
#include <thrust/binary_search.h>
#include <iostream>

template <typename Vector>
void print_vector(const std::string& name, const Vector& v)
{
  typedef typename Vector::value_type T;
  std::cout << "  "  << name << "  ";
  thrust::copy(v.begin(), v.end(), std::ostream_iterator<T>(std::cout, " "));
  std::cout << std::endl;
}


template <typename Vector1, 
          typename Vector2>
void dense_histogram(const Vector1& input,
                           Vector2& histogram)
{
  typedef typename Vector1::value_type ValueType; // input value type
  typedef typename Vector2::value_type IndexType; // histogram index type

  // copy input data (could be skipped if input is allowed to be modified)
  thrust::device_vector<ValueType> data(input);
    
  // print the initial data
  print_vector("initial data", data);

  // sort data to bring equal elements together
  thrust::sort(data.begin(), data.end());
  
  // print the sorted data
  print_vector("sorted data", data);

  // number of histogram bins is equal to the maximum value plus one
  IndexType num_bins = data.back() + 1;

  // resize histogram storage
  histogram.resize(num_bins);
  
  // find the end of each bin of values
  thrust::counting_iterator<IndexType> search_begin(0);
  thrust::upper_bound(data.begin(), data.end(),
                      search_begin, search_begin + num_bins,
                      histogram.begin());
  
  // print the cumulative histogram
  print_vector("cumulative histogram", histogram);

  // compute the histogram by taking differences of the cumulative histogram
  //thrust::adjacent_difference(histogram.begin(), histogram.end(),
    //                          histogram.begin());

  // print the histogram
  //print_vector("histogram", histogram);
}


int main()
{
    thrust::default_random_engine rng;
    thrust::uniform_int_distribution<int> dist(0, 9);

    const int N = 40;
    const int S = 4;
    
    // generate random data on the host
    thrust::host_vector<int> input(N);
    for(int i = 0; i < N; i++)
    {
        int sum = 0;
        for (int j = 0; j < S; j++)
          sum += dist(rng);
        input[i] = sum / S;
    }
    print_vector("Input",input);
    thrust::device_vector<int> histogram;
    dense_histogram(input, histogram);
    return 0;
}