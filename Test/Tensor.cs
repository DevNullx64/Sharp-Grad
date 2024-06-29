using ILGPU.Algorithms;
using ILGPU.Util;
using SharpGrad;
using SharpGrad.Tensors;
using System.Diagnostics;
using System.Numerics;
using System.Security.Cryptography;

namespace Test
{
    [TestClass]

    public class Operators<T>
        where T : unmanaged, IFloatingPoint<T>, IPowerFunctions<T>, IExponentialFunctions<T>, ILogarithmicFunctions<T>
    {
        static readonly Random rnd = new();

        public static T Epsilon = T.CreateChecked(1e-6);
        public static void Fill(TensorData<T> result, Func<int, int, int, T> fnc)
        {
            long begin = DateTime.Now.Ticks;
            for (int i = 0; i < result.Shape[0]; i++)
                for (int j = 0; j < result.Shape[1]; j++)
                    for (int k = 0; k < result.Shape[2]; k++)
                        result.Set(fnc(i, j, k), i, j, k);
            Debug.WriteLine($"Fill of {result.Name} took {(DateTime.Now.Ticks - begin) / 10000} ms");
        }

        public static TensorData<T> NewRandom(params Dim[] dims)
        {
            long begin = DateTime.Now.Ticks;
            TensorData<T> result = new(dims);
            for (int i = 0; i < dims[0]; i++)
                for (int j = 0; j < dims[1]; j++)
                    for (int k = 0; k < dims[2]; k++)
                        result.Set(T.CreateTruncating((rnd.NextDouble() + 1) * 2), i, j, k);
            Debug.WriteLine($"NewRandom of {result.Name} took {(DateTime.Now.Ticks - begin) / 10000} ms");
            return result;
        }

        public static (T Mean, T Min, T Max) Test(Tensor<T> tc, Tensor<T> ty)
        {
            long begin = DateTime.Now.Ticks;
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
            T factor = T.CreateTruncating(tc.Shape.Length);
            Debug.WriteLine($"Test between {nameof(tc)} and {nameof(ty)} took {(DateTime.Now.Ticks - begin) / 10000} ms");
            return (factor == T.Zero ? diff : diff / factor, min, max);
        }

        public static void Addition()
        {
            KernelProcessUnit kpu = KernelProcessUnit.DefaultKPU;
            kpu.PrintInformation(Console.Out);

            Tensor<T> ta = NewRandom(256, 256, 256);
            Tensor<T> tb = NewRandom(256, 256, 256);

            TensorData<T> ty = (nameof(ty), new(256, 256, 256));
            Fill(ty, (d, i, j) => ta[d, i, j] + tb[d, i, j]);

            Tensor<T> tc = ta + tb;

            long begin = DateTime.Now.Ticks;
            _ = tc[0, 0, 0];
            Debug.WriteLine($"Get result of {nameof(tc)} takes {(DateTime.Now.Ticks - begin) / 10000} ms");

            (T mean, T min, T max) = Test(tc, ty);
            Assert.IsTrue(mean <= Epsilon && min <= Epsilon && max <= Epsilon, $"mean={mean}/0, min={min}/0, max={max}/0");
            Debug.WriteLine($"Addition test passed with error mean={mean}, min={min}, max={max}");
        }
        public static void Subtraction()
        {
            KernelProcessUnit kpu = KernelProcessUnit.DefaultKPU;
            kpu.PrintInformation(Console.Out);

            Tensor<T> ta = NewRandom(256, 256, 256);
            Tensor<T> tb = NewRandom(256, 256, 256);

            TensorData<T> ty = (nameof(ty), new(256, 256, 256));
            Fill(ty, (d, i, j) => ta[d, i, j] - tb[d, i, j]);

            Tensor<T> tc = ta - tb;

            long begin = DateTime.Now.Ticks;
            _ = tc[0, 0, 0];
            Debug.WriteLine($"Get result of {nameof(tc)} takes {(DateTime.Now.Ticks - begin) / 10000} ms");

            (T mean, T min, T max) = Test(tc, ty);
            Assert.IsTrue(mean <= Epsilon && min <= Epsilon && max <= Epsilon, $"mean={mean}/0, min={min}/0, max={max}/0");
            Debug.WriteLine($"Subtraction test passed with error mean={mean}, min={min}, max={max}");
        }

        public static void Multiplication()
        {
            KernelProcessUnit kpu = KernelProcessUnit.DefaultKPU;
            kpu.PrintInformation(Console.Out);

            Tensor<T> ta = NewRandom(256, 256, 256);
            Tensor<T> tb = NewRandom(256, 256, 256);

            TensorData<T> ty = (nameof(ty), new(256, 256, 256));
            Fill(ty, (d, i, j) => ta[d, i, j] * tb[d, i, j]);

            Tensor<T> tc = ta * tb;

            long begin = DateTime.Now.Ticks;
            _ = tc[0, 0, 0];
            Debug.WriteLine($"Get result of {nameof(tc)} takes {(DateTime.Now.Ticks - begin) / 10000} ms");

            (T mean, T min, T max) = Test(tc, ty);
            Assert.IsTrue(mean <= Epsilon && min <= Epsilon && max <= Epsilon, $"mean={mean}/0, min={min}/0, max={max}/0");
            Debug.WriteLine($"Multiplication test passed with error mean={mean}, min={min}, max={max}");
        }

        public static void Division()
        {
            KernelProcessUnit kpu = KernelProcessUnit.DefaultKPU;
            kpu.PrintInformation(Console.Out);

            Tensor<T> ta = NewRandom(256, 256, 256);
            Tensor<T> tb = NewRandom(256, 256, 256);

            TensorData<T> ty = new TensorData<T>(nameof(ty), new(256, 256, 256));
            Fill(ty, (d, i, j) => ta[d, i, j] / tb[d, i, j]);

            Tensor<T> tc = ta / tb;

            long begin = DateTime.Now.Ticks;
            _ = tc[0, 0, 0];
            Debug.WriteLine($"Get result of {nameof(tc)} takes {(DateTime.Now.Ticks - begin) / 10000} ms");

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

    /*
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
    */

    /*
    [TestClass]
    public class CastTestOperatorsULong : Operators<double>
    {
        private bool Test<T>(Tensor<T> tensor)
            where T : unmanaged, INumber<T>
        {
            for(int i =0;i < tensor.Shape[0]; i++)
                for(int j = 0; j < tensor.Shape[1]; j++)
                    for(int k = 0; k < tensor.Shape[2]; k++)
                        if(tensor[i, j, k] != T.One)
                            return false;
            return true;
        }
        [TestMethod]
        public void CastDoubleToFloat()
        {
            Tensor<double> ta = NewRandom(256, 256, 256);
            Fill(ta, (i, j, k) => 1);
            Tensor<double> tDouble = Tensor<double>.CastTo<double>(ta);
            Assert.IsTrue(Test(tDouble), $"double to double casting failed");
        }
        [TestMethod]
        public void CastDoubleToULong()
        {
            Tensor<double> ta = NewRandom(256, 256, 256);
            Fill(ta, (i, j, k) => 1);
            Tensor<ulong> tULong = Tensor<double>.CastTo<ulong>(ta);
            Assert.IsTrue(Test(tULong), $"double to ulong casting failed");
        }
        [TestMethod]
        public void CastDoubleToLong()
        {
            Tensor<double> ta = NewRandom(256, 256, 256);
            Fill(ta, (i, j, k) => 1);
            Tensor<long> tLong = Tensor<double>.CastTo<long>(ta);
            Assert.IsTrue(Test(tLong), $"double to long casting failed");
        }
        [TestMethod]
        public void CastDoubleToUInt()
        {
            Tensor<double> ta = NewRandom(256, 256, 256);
            Fill(ta, (i, j, k) => 1);
            Tensor<uint> tUInt = Tensor<double>.CastTo<uint>(ta);
            Assert.IsTrue(Test(tUInt), $"double to uint casting failed");
        }
        [TestMethod]
        public void CastDoubleToInt()
        {
            Tensor<double> ta = NewRandom(256, 256, 256);
            Fill(ta, (i, j, k) => 1);
            Tensor<int> tInt = Tensor<double>.CastTo<int>(ta);
            Assert.IsTrue(Test(tInt), $"double to int casting failed");
        }
        [TestMethod]
        public void CastDoubleToUShort()
        {
            Tensor<double> ta = NewRandom(256, 256, 256);
            Fill(ta, (i, j, k) => 1);
            Tensor<ushort> tUShort = Tensor<double>.CastTo<ushort>(ta);
            Assert.IsTrue(Test(tUShort), $"double to ushort casting failed");
        }
        [TestMethod]
        public void CastDoubleToShort()
        {
            Tensor<double> ta = NewRandom(256, 256, 256);
            Fill(ta, (i, j, k) => 1);
            Tensor<short> tShort = Tensor<double>.CastTo<short>(ta);
            Assert.IsTrue(Test(tShort), $"double to short casting failed");
        }
        [TestMethod]
        public void CastDoubleToByte()
        {
            Tensor<double> ta = NewRandom(256, 256, 256);
            Fill(ta, (i, j, k) => 1);
            Tensor<byte> tByte = Tensor<double>.CastTo<byte>(ta);
            Assert.IsTrue(Test(tByte), $"double to byte casting failed");
        }
        [TestMethod]
        public void CastDoubleToSByte()
        {
            Tensor<double> ta = NewRandom(256, 256, 256);
            Fill(ta, (i, j, k) => 1);
            Tensor<sbyte> tSByte = Tensor<double>.CastTo<sbyte>(ta);
            Assert.IsTrue(Test(tSByte), $"double to sbyte casting failed");
        }
    }
    */
}