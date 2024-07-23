using ILGPU.Algorithms;
using SharpGrad;
using SharpGrad.Tensors;
using System.Diagnostics;
using System.Numerics;
using System.Security.Cryptography;
using SharpGrad.Tensors.Operators;

namespace Test
{
    [TestClass]
    public class Computation
    {

        // Y = (A + B) ∗ (B − C)
        private void TestComputation<T>()
            where T : unmanaged, INumber<T>, IFloatingPoint<T>, IPowerFunctions<T>, IExponentialFunctions<T>, ILogarithmicFunctions<T>
        {
            //T epsilon = T.CreateChecked(1e-5);
            TensorData<T> A = Operators<T>.NewRandom(256, 256, 256);
            TensorData<T> B = Operators<T>.NewRandom(256, 256, 256);
            TensorData<T> C = Operators<T>.NewRandom(256, 256, 256);
            TensorData<T> Y = new("Y", new(256, 256, 256));

            Operators<T>.Fill(Y, (i, j, k) =>
            {
                T a = A[i, j, k];
                T b = B[i, j, k];
                T c = C[i, j, k];
                T d = (a + b) * (b - c);
                T e = (a - c) / (b * c);
                return d / e;
            });

            Tensor<T> D = (A + B) * (B - C);
            Tensor<T> E = (A - C) / (B * C);
            var cY = D / E;

            long begin = DateTime.Now.Ticks;
            _ = cY[0, 0, 0];
            Debug.WriteLine($"Get result of {nameof(cY)} takes {(DateTime.Now.Ticks - begin) / 10000} ms");

            (T mean, T min, T max) = Operators<T>.Test(Y, cY);

            Assert.IsTrue(mean <= Operators<T>.Epsilon && min <= Operators<T>.Epsilon && max <= Operators<T>.Epsilon, $"mean={mean}/0, min={min}/0, max={max}/0");
            Debug.WriteLine($"Computation test passed with error mean={mean}, min={min}, max={max}");
        }

        private void TestComputeAndReduce<T>(Shape shape, int dim = 2)
            where T : unmanaged, INumber<T>, IFloatingPoint<T>, IPowerFunctions<T>, IExponentialFunctions<T>, ILogarithmicFunctions<T>
        {
            //T epsilon = T.CreateChecked(1e-5);
            TensorData<T> A = new("A", shape);
            Operators<T>.Fill(A, (i, j, k) => T.CreateChecked(((i * shape[0] + j) * shape[1]) + k));

            TensorData<T> B = new("B", shape);
            Operators<T>.Fill(B, (i, j, k) => T.CreateChecked(k));

            TensorData<T> Y = new("Y", new(shape[0], shape[1], shape[2]));

            Operators<T>.Fill(Y, (i, j, k) =>
            {
                return A[i, j, k] * B[i, j, k];
            });
            Y = Operators<T>.Reduce<AddOp<T>>(Y, dim);
            Tensor<T> C = A * B;

            long begin = DateTime.Now.Ticks;
            Tensor<T> cY = C.Sum(dim);
            _ = cY[0, 0, 0];
            Debug.WriteLine($"Get result of {nameof(cY)} takes {(DateTime.Now.Ticks - begin) / 10000} ms");

            (T mean, T min, T max) = Operators<T>.Test(Y, cY);

            Assert.IsTrue(mean <= Operators<T>.Epsilon && min <= Operators<T>.Epsilon && max <= Operators<T>.Epsilon, $"mean={mean}/0, min={min}/0, max={max}/0");
            Debug.WriteLine($"Compute & Reduce test passed with error mean={mean}, min={min}, max={max}");
        }

        [TestMethod]
        public void FloatTestComputation()
        {
            TestComputation<float>();
        }

        [TestMethod]
        public void DoubleTestComputation()
        {
            TestComputation<double>();
        }

        private void TestComputeAndReduce<T>(Shape shape)
            where T : unmanaged, INumber<T>, IFloatingPoint<T>, IPowerFunctions<T>, IExponentialFunctions<T>, ILogarithmicFunctions<T>
        {
            Debug.WriteLine($"Test shape {shape}");
            for (int i = 2; i >= 0; i--)
            {
                TestComputeAndReduce<T>(shape, i);
                Debug.WriteLine($"\nTestComputeAndReduce<{typeof(T).Name}>({shape}, {i}) finished");
            }
        }

        private void TestComputeAndReduce<T>()
        where T : unmanaged, INumber<T>, IFloatingPoint<T>, IPowerFunctions<T>, IExponentialFunctions<T>, ILogarithmicFunctions<T>
        {
            TestComputeAndReduce<T>(new(2, 2, 2));
            TestComputeAndReduce<T>(new(31, 31, 31));
            TestComputeAndReduce<T>(new(32, 32, 32));
            TestComputeAndReduce<T>(new(33, 33, 33));
            TestComputeAndReduce<T>(new(31, 32, 33));

            Debug.WriteLine($"\nTestComputeAndReduce<{typeof(T).Name}> random shape test");
            for(int i = 0; i < 3; i++)
                TestComputeAndReduce<T>(new(Random.Shared.Next(1, 256), Random.Shared.Next(1, 256), Random.Shared.Next(1, 256)));

            Debug.WriteLine($"\nTestComputeAndReduce<{typeof(T).Name}> finished");
        }


        [TestMethod]
        public void FloatTestComputeAndReduce()
        {
            TestComputeAndReduce<float>();
        }

        [TestMethod]
        public void DoubleTestComputeAndReduce()
        {
            TestComputeAndReduce<double>();
        }
    }
}