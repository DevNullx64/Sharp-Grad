using ILGPU.Algorithms;
using SharpGrad;
using SharpGrad.Tensors;
using System.Diagnostics;
using System.Numerics;
using System.Security.Cryptography;

namespace Test
{
    [TestClass]

    public class Operators<T>
        where T : unmanaged, IFloatingPoint<T>
    {
        static readonly Random rnd = new();

        public static T Epsilon = T.CreateChecked(1e-6);
        public static void Fill(Tensor<T> result, Func<int, int, int, T> fnc)
        {
            for (int i = 0; i < result.Shape[0]; i++)
                for (int j = 0; j < result.Shape[1]; j++)
                    for (int k = 0; k < result.Shape[2]; k++)
                        result[i, j, k] = fnc(i, j, k);
        }

        public static Tensor<T> NewRandom(params Dim[] dims)
        {
            Tensor<T> result = new(dims);
            for (int i = 0; i < dims[0]; i++)
                for (int j = 0; j < dims[1]; j++)
                    for (int k = 0; k < dims[2]; k++)
                        result[i, j, k] = T.CreateChecked(rnd.NextDouble() * 100);
            return result;
        }

        public static (T Mean, T Min, T Max) Test(Tensor<T> tc, Tensor<T> ty)
        {
            T diff = T.Zero;
            T min = T.CreateChecked(double.MaxValue);
            T max = T.CreateChecked(double.MinValue);

            for (int d = 0; d < tc.Shape[0]; d++)
                for (int i = 0; i < tc.Shape[1]; i++)
                    for (int j = 0; j < tc.Shape[2]; j++)
                    {
                        T diff_ = tc[d, i, j] - ty[d, i, j];
                        if (diff_ < T.Zero)
                            diff_ *= T.NegativeOne;

                        diff += diff_;
                        if (diff_ < min)
                            min = diff_;
                        if (diff_ > max)
                            max = diff_;
                    }
            return (diff / T.CreateChecked(tc.Shape.Size), min, max);
        }

        [TestMethod]
        public void TestKPU()
        {
            // Test KPU
            //OperationKPU[] ops =
            //    [
            //        new(OpCode.Add, 0, 1, -1),
            //        new(OpCode.Mul, -1, 1, -1),
            //        new(OpCode.Add, 0, -1, -1),
            //        new(OpCode.Save, -1, -1, 2)
            //    ];
            //Fill(ty, (i, j, k) =>
            //{
            //    float value = ta[i, j, k] + tb[i, j, k];
            //    value *= tb[i, j, k];
            //    value += ta[i, j, k];
            //    return value;
            //});
            //tc = Tensor<float>.ExecGpu(ops, ta, tb, tc);

        }

        [TestMethod]
        public static void Dynamic()
        {
            Tensors.Accelerator.PrintInformation(Console.Out);

            Tensor<T> ta = NewRandom(256, 256, 256);
            Tensor<T> tb = NewRandom(256, 256, 256);
            Tensor<T> tc = new(256, 256, 256);
            Tensor<T> ty = new(256, 256, 256);

            // Test dynamic operations
            Fill(ty, (i, j, k) =>
            {
                ty[i, j, k] += ta[i, j, k] - tb[i, j, k];
                ty[i, j, k] += ta[i, j, k] + tb[i, j, k];
                ty[i, j, k] += ta[i, j, k] * tb[i, j, k];
                ty[i, j, k] += ta[i, j, k] / tb[i, j, k];
                return ty[i, j, k];
            });
            Tensor<T>.DynGpu([OpCode.Sub, OpCode.Add, OpCode.Mul, OpCode.Div], ta, tb, tc);

            (T mean, T min, T max) = Test(tc, ty);
            Assert.IsTrue(mean <= Epsilon && min <= Epsilon && max <= Epsilon, $"mean={mean}/0, min={min}/0, max={max}/0");
            Debug.WriteLine($"dynamic test passed for type {typeof(T).Name} with error mean={mean}, min={min}, max={max}");
        }

        public static void Addition()
        {
            Tensors.Accelerator.PrintInformation(Console.Out);

            Tensor<T> ta = NewRandom(256, 256, 256);
            Tensor<T> tb = NewRandom(256, 256, 256);
            Tensor<T> tc = new(256, 256, 256);

            Tensor<T> ty = new(256, 256, 256);
            Fill(ty, (d, i, j) => ta[d, i, j] + tb[d, i, j]);

            tc = ta + tb;

            (T mean, T min, T max) = Test(tc, ty);
            Assert.IsTrue(mean <= Epsilon && min <= Epsilon && max <= Epsilon, $"mean={mean}/0, min={min}/0, max={max}/0");
            Debug.WriteLine($"Addition test passed with error mean={mean}, min={min}, max={max}");
        }
        public static void Subtraction()
        {
            Tensors.Accelerator.PrintInformation(Console.Out);

            Tensor<T> ta = NewRandom(256, 256, 256);
            Tensor<T> tb = NewRandom(256, 256, 256);
            Tensor<T> tc = new(256, 256, 256);

            Tensor<T> ty = new(256, 256, 256);
            Fill(ty, (d, i, j) => ta[d, i, j] - tb[d, i, j]);

            tc = ta - tb;

            (T mean, T min, T max) = Test(tc, ty);
            Assert.IsTrue(mean <= Epsilon && min <= Epsilon && max <= Epsilon, $"mean={mean}/0, min={min}/0, max={max}/0");
            Debug.WriteLine($"Subtraction test passed with error mean={mean}, min={min}, max={max}");
        }
        public static void Multiplication()
        {
            Tensors.Accelerator.PrintInformation(Console.Out);

            Tensor<T> ta = NewRandom(256, 256, 256);
            Tensor<T> tb = NewRandom(256, 256, 256);
            Tensor<T> tc = new(256, 256, 256);

            Tensor<T> ty = new(256, 256, 256);
            Fill(ty, (d, i, j) => ta[d, i, j] * tb[d, i, j]);

            tc = ta * tb;

            (T mean, T min, T max) = Test(tc, ty);
            Assert.IsTrue(mean <= Epsilon && min <= Epsilon && max <= Epsilon, $"mean={mean}/0, min={min}/0, max={max}/0");
            Debug.WriteLine($"Multiplication test passed with error mean={mean}, min={min}, max={max}");
        }
        public static void Division()
        {
            Tensors.Accelerator.PrintInformation(Console.Out);

            Tensor<T> ta = NewRandom(256, 256, 256);
            Tensor<T> tb = NewRandom(256, 256, 256);
            Tensor<T> tc = new(256, 256, 256);

            Tensor<T> ty = new(256, 256, 256);
            Fill(ty, (d, i, j) => ta[d, i, j] / tb[d, i, j]);

            tc = ta / tb;

            (T mean, T min, T max) = Test(tc, ty);
            Assert.IsTrue(mean <= Epsilon && min <= Epsilon && max <= Epsilon, $"mean={mean}/0, min={min}/0, max={max}/0");
            Debug.WriteLine($"Division test passed with error mean={mean}, min={min}, max={max}");
        }
    }
    /*
    [TestClass]
    public class OperatorsHalf : Operators<ILGPU.Half>
    {
        [TestMethod]
        public void TestAddition() => Addition();
        [TestMethod]
        public void TestSubtraction() => Subtraction();
        [TestMethod]
        public void TestMultiplication() => Multiplication();
        [TestMethod]
        public void TestDivision() => Division();
        [TestMethod]
        public void TestDynamic() => Dynamic();
    }
    */

    [TestClass]
    public class OperatorsFloat : Operators<float>
    {
        [TestMethod]
        public void TestAddition() => Addition();
        [TestMethod]
        public void TestSubtraction() => Subtraction();
        [TestMethod]
        public void TestMultiplication() => Multiplication();
        [TestMethod]
        public void TestDivision() => Division();
        [TestMethod]
        public void TestDynamic() => Dynamic();
    }

    [TestClass]
    public class OperatorsDouble : Operators<double>
    {
        [TestMethod]
        public void TestAddition() => Addition();
        [TestMethod]
        public void TestSubtraction() => Subtraction();
        [TestMethod]
        public void TestMultiplication() => Multiplication();
        [TestMethod]
        public void TestDivision() => Division();
        [TestMethod]
        public void TestDynamic() => Dynamic();
    }
}