using ILGPU.Algorithms;
using SharpGrad;
using SharpGrad.Tensors;
using System.Diagnostics;
using System.Numerics;
using System.Security.Cryptography;

namespace Test
{
    [TestClass]
    public class Computation
    {

        // Y = (A + B) ∗ (B − C)
        private void TestComputation<T>()
            where T: unmanaged, INumber<T>, IFloatingPoint<T>, IPowerFunctions<T>, IExponentialFunctions<T>, ILogarithmicFunctions<T>
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
            Debug.WriteLine($"Multiplication test passed with error mean={mean}, min={min}, max={max}");
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
    }
}