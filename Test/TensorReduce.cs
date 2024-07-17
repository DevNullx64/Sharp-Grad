using SharpGrad;
using SharpGrad.Tensors;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Numerics;
using System.Text;
using System.Threading.Tasks;

namespace Test
{
    [TestClass]
    public class TensorReduce
    {
        public static readonly Shape TestShape = new(256, 256, 256);
        [TestMethod]
        public void TestReduceSum()
        {
            Tensor<double> a = Operators<double>.NewRandom(TestShape);

            long start = DateTime.Now.Ticks;
            TensorData<double> ty = new(TestShape.SetDim(^1, 1));
            for (int i = 0; i < TestShape[0]; i++)
                for (int j = 0; j < TestShape[1]; j++)
                {
                    ty.Set(0, i, j, 0);
                    for (int k = 0; k < TestShape[2]; k++)
                        ty.Set(ty[i, j, 0] + a[i, j, k], i, j, 0);
                }
            long end = DateTime.Now.Ticks;
            Debug.WriteLine($"C# for loop: {(end - start) / 10000} ms");

            start = DateTime.Now.Ticks;
            Tensor<double> sum = a.Sum();
            end = DateTime.Now.Ticks;
            Debug.WriteLine($"SharpGrad: {(end - start) / 10000} ms");

            for (int i = 0; i < TestShape[0]; i++)
                for (int j = 0; j < TestShape[1]; j++)
                {
                    double d = Math.Abs(sum[i, j, 0] - ty[i, j, 0]);
                    Assert.IsTrue(d < 1e-2, $"Error [{i},{j}]: {d} > 1e-2");
                }
        }

        [TestMethod]
        public void TestReduceSum2()
        {
            Tensor<double> a = Operators<double>.NewRandom(TestShape);

            long start = DateTime.Now.Ticks;
            TensorData<double> ty = new(TestShape.SetDim(^1, 1));
            for (int i = 0; i < TestShape[0]; i++)
                for (int j = 0; j < TestShape[1]; j++)
                {
                    ty.Set(0, i, j, 0);
                    for (int k = 0; k < TestShape[2]; k++)
                        ty.Set(ty[i, j, 0] + a[i, j, k], i, j, 0);
                }
            long end = DateTime.Now.Ticks;
            Debug.WriteLine($"C# for loop: {(end - start) / 10000} ms");

            start = DateTime.Now.Ticks;
            Tensor<double> sum = a.Sum();
            end = DateTime.Now.Ticks;
            Debug.WriteLine($"SharpGrad: {(end - start) / 10000} ms");

            for (int i = 0; i < TestShape[0]; i++)
                for (int j = 0; j < TestShape[1]; j++)
                {
                    double d = Math.Abs(sum[i, j, 0] - ty[i, j, 0]);
                    Assert.IsTrue(d < 1e-2, $"Error [{i},{j}]: {d} > 1e-2");
                }
        }
    }
}
