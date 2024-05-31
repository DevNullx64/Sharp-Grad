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
        where T : unmanaged, INumber<T>
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

        public static DataTensor<T> NewRandom(params Dim[] dims)
        {
            DataTensor<T> result = new(dims);
            for (int i = 0; i < dims[0]; i++)
                for (int j = 0; j < dims[1]; j++)
                    for (int k = 0; k < dims[2]; k++)
                        result[i, j, k] = T.CreateTruncating((rnd.NextDouble() + 1) * 2);
            return result;
        }

        public static (T Mean, T Min, T Max) Test(Tensor<T> tc, Tensor<T> ty)
        {
            T diff = T.Zero;
            T min = T.CreateTruncating(double.MaxValue);
            T max = T.CreateTruncating(double.MinValue);

            for (int d = 0; d < tc.Shape[0]; d++)
                for (int i = 0; i < tc.Shape[1]; i++)
                    for (int j = 0; j < tc.Shape[2]; j++)
                    {
                        T diff_ = tc[d, i, j] - ty[d, i, j];
                        if (diff_ < T.Zero)
                            diff_ *= -T.Zero;

                        diff += diff_;
                        if (diff_ < min)
                            min = diff_;
                        if (diff_ > max)
                            max = diff_;
                    }
            T factor = T.CreateTruncating(tc.Shape.Size);
            return (factor == T.Zero ? diff : diff / factor, min, max);
        }

        public static void Addition()
        {
            Acc.Accelerator.PrintInformation(Console.Out);

            Tensor<T> ta = NewRandom(256, 256, 256);
            Tensor<T> tb = NewRandom(256, 256, 256);

            DataTensor<T> ty = new(256, 256, 256);
            Fill(ty, (d, i, j) => ta[d, i, j] + tb[d, i, j]);

            Tensor<T> tc = ta + tb;

            (T mean, T min, T max) = Test(tc, ty);
            Assert.IsTrue(mean <= Epsilon && min <= Epsilon && max <= Epsilon, $"mean={mean}/0, min={min}/0, max={max}/0");
            Debug.WriteLine($"Addition test passed with error mean={mean}, min={min}, max={max}");
        }
        public static void Subtraction()
        {
            Acc.Accelerator.PrintInformation(Console.Out);

            Tensor<T> ta = NewRandom(256, 256, 256);
            Tensor<T> tb = NewRandom(256, 256, 256);

            Tensor<T> ty = new DataTensor<T>(256, 256, 256);
            Fill(ty, (d, i, j) => ta[d, i, j] - tb[d, i, j]);

            Tensor<T> tc = ta - tb;

            (T mean, T min, T max) = Test(tc, ty);
            Assert.IsTrue(mean <= Epsilon && min <= Epsilon && max <= Epsilon, $"mean={mean}/0, min={min}/0, max={max}/0");
            Debug.WriteLine($"Subtraction test passed with error mean={mean}, min={min}, max={max}");
        }
        public static void Multiplication()
        {
            Acc.Accelerator.PrintInformation(Console.Out);

            Tensor<T> ta = NewRandom(256, 256, 256);
            Tensor<T> tb = NewRandom(256, 256, 256);

            Tensor<T> ty = new DataTensor<T>(256, 256, 256);
            Fill(ty, (d, i, j) => ta[d, i, j] * tb[d, i, j]);

            Tensor<T> tc = ta * tb;

            (T mean, T min, T max) = Test(tc, ty);
            Assert.IsTrue(mean <= Epsilon && min <= Epsilon && max <= Epsilon, $"mean={mean}/0, min={min}/0, max={max}/0");
            Debug.WriteLine($"Multiplication test passed with error mean={mean}, min={min}, max={max}");
        }
        public static void Division()
        {
            Acc.Accelerator.PrintInformation(Console.Out);

            Tensor<T> ta = NewRandom(256, 256, 256);
            Tensor<T> tb = NewRandom(256, 256, 256);

            Tensor<T> ty = new DataTensor<T>(256, 256, 256);
            Fill(ty, (d, i, j) => ta[d, i, j] / tb[d, i, j]);

            Tensor<T> tc = ta / tb;

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
    }

    [TestClass]
    public class OperatorsByte : Operators<byte>
    {
        [TestMethod]
        public void TestAddition() => Addition();
        [TestMethod]
        public void TestSubtraction() => Subtraction();
        [TestMethod]
        public void TestMultiplication() => Multiplication();
        [TestMethod]
        public void TestDivision() => Division();
    }

    [TestClass]
    public class OperatorsSByte : Operators<sbyte>
    {
        [TestMethod]
        public void TestAddition() => Addition();
        [TestMethod]
        public void TestSubtraction() => Subtraction();
        [TestMethod]
        public void TestMultiplication() => Multiplication();
        [TestMethod]
        public void TestDivision() => Division();
    }

    [TestClass]
    public class OperatorsShort : Operators<short>
    {
        [TestMethod]
        public void TestAddition() => Addition();
        [TestMethod]
        public void TestSubtraction() => Subtraction();
        [TestMethod]
        public void TestMultiplication() => Multiplication();
        [TestMethod]
        public void TestDivision() => Division();
    }

    [TestClass]
    public class OperatorsUShort : Operators<ushort>
    {
        [TestMethod]
        public void TestAddition() => Addition();
        [TestMethod]
        public void TestSubtraction() => Subtraction();
        [TestMethod]
        public void TestMultiplication() => Multiplication();
        [TestMethod]
        public void TestDivision() => Division();
    }

    [TestClass]
    public class OperatorsInt : Operators<int>
    {
        [TestMethod]
        public void TestAddition() => Addition();
        [TestMethod]
        public void TestSubtraction() => Subtraction();
        [TestMethod]
        public void TestMultiplication() => Multiplication();
        [TestMethod]
        public void TestDivision() => Division();
    }

    [TestClass]
    public class OperatorsUInt : Operators<uint>
    {
        [TestMethod]
        public void TestAddition() => Addition();
        [TestMethod]
        public void TestSubtraction() => Subtraction();
        [TestMethod]
        public void TestMultiplication() => Multiplication();
        [TestMethod]
        public void TestDivision() => Division();
    }

    [TestClass]
    public class OperatorsLong : Operators<long>
    {
        [TestMethod]
        public void TestAddition() => Addition();
        [TestMethod]
        public void TestSubtraction() => Subtraction();
        [TestMethod]
        public void TestMultiplication() => Multiplication();
        [TestMethod]
        public void TestDivision() => Division();
    }

    [TestClass]
    public class OperatorsULong : Operators<ulong>
    {
        [TestMethod]
        public void TestAddition() => Addition();
        [TestMethod]
        public void TestSubtraction() => Subtraction();
        [TestMethod]
        public void TestMultiplication() => Multiplication();
        [TestMethod]
        public void TestDivision() => Division();
    }
}