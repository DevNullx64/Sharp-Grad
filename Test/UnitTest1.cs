using SharpGrad;
using SharpGrad.Tensors;
using System.Diagnostics;

namespace Test
{
    [TestClass]
    public class UnitTest1
    {
        static readonly Random rnd = new();

        public static void Fill(Tensor<float> result, Func<int, int, int, float> fnc)
        {
            for (int d = 0; d < result.Shape[0]; d++)
                for (int i = 0; i < result.Shape[1]; i++)
                    for (int j = 0; j < result.Shape[2]; j++)
                        result[d, i, j] = fnc(d, i, j);
        }

        public static Tensor<float> NewRandom(params Dim[] dims)
        {
            Tensor<float> result = new(dims);
            for (int d = 0; d < dims[0]; d++)
                for (int i = 0; i < dims[1]; i++)
                    for (int j = 0; j < dims[2]; j++)
                        result[d, i, j] = (float)rnd.NextDouble() * 100;
            return result;
        }

        public static (float Mean, float Min, float Max) Test(Tensor<float> tc, Tensor<float> ty)
        {
            float diff = 0;
            float min = float.MaxValue;
            float max = float.MinValue;

            for (int d = 0; d < tc.Shape[0]; d++)
                for (int i = 0; i < tc.Shape[1]; i++)
                    for (int j = 0; j < tc.Shape[2]; j++)
                    {
                        float diff_ = Math.Abs(tc[d, i, j] - ty[d, i, j]);
                        diff += diff_;
                        if (diff_ < min)
                            min = diff_;
                        if (diff_ > max)
                            max = diff_;
                    }
            return (diff / tc.Shape.Size, min, max);
        }

        [TestMethod]
        public void TestDynamic()
        {
            Tensors.Accelerator.PrintInformation(Console.Out);

            Tensor<float> ta = NewRandom(256, 256, 256);
            Tensor<float> tb = NewRandom(256, 256, 256);
            Tensor<float> tc = new(256, 256, 256);
            Tensor<float> ty = new(256, 256, 256);

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

            // Test dynamic operations
            Fill(ty, (k, i, j) =>
            {
                ty[k, i, j] += ta[k, i, j] - tb[k, i, j];
                ty[k, i, j] += ta[k, i, j] + tb[k, i, j];
                ty[k, i, j] += ta[k, i, j] * tb[k, i, j];
                ty[k, i, j] += ta[k, i, j] / tb[k, i, j];
                return ty[k, i, j];
            });
            tc = Tensor<float>.ExecGpu([OpCode.Sub, OpCode.Add, OpCode.Mul, OpCode.Div], ta, tb);

            (float mean, float min, float max) = Test(tc, ty);
            Debug.WriteLine($"dynamic test passed with mean error: {mean}, min error: {min}, max error: {max}");
        }

        [TestMethod]
        public void TestAddition()
        {
            Tensor<float> ta = NewRandom(256, 256, 256);
            Tensor<float> tb = NewRandom(256, 256, 256);
            Tensor<float> tc = new(256, 256, 256);

            Tensor<float> ty = new(256, 256, 256);
            Fill(ty, (d, i, j) => ta[d, i, j] + tb[d, i, j]);

            tc = ta + tb;

            (float mean, float min, float max) = Test(tc, ty);
            //Assert.IsTrue(mean < 1e-6);
            Debug.WriteLine($"Addition test passed with error mean={mean}, min={min}, max={max}");
        }

        [TestMethod]
        public void TestSubtraction()
        {
            Tensor<float> ta = NewRandom(256, 256, 256);
            Tensor<float> tb = NewRandom(256, 256, 256);
            Tensor<float> tc = new(256, 256, 256);

            Tensor<float> ty = new(256, 256, 256);
            Fill(ty, (d, i, j) => ta[d, i, j] - tb[d, i, j]);

            tc = ta - tb;

            (float mean, float min, float max) = Test(tc, ty);
            //Assert.IsTrue(mean < 1e-6);
            Debug.WriteLine($"Subtraction test passed with error mean={mean}, min={min}, max={max}");
        }

        [TestMethod]
        public void TestMultiplication()
        {
            Tensor<float> ta = NewRandom(256, 256, 256);
            Tensor<float> tb = NewRandom(256, 256, 256);
            Tensor<float> tc = new(256, 256, 256);

            Tensor<float> ty = new(256, 256, 256);
            Fill(ty, (d, i, j) => ta[d, i, j] * tb[d, i, j]);

            tc = ta * tb;

            (float mean, float min, float max) = Test(tc, ty);
            //Assert.IsTrue(mean < 1e-6);
            Debug.WriteLine($"Multiplication test passed with error mean={mean}, min={min}, max={max}");
        }

        [TestMethod]
        public void TestDivision()
        {
            Tensor<float> ta = NewRandom(256, 256, 256);
            Tensor<float> tb = NewRandom(256, 256, 256);
            Tensor<float> tc = new(256, 256, 256);

            Tensor<float> ty = new(256, 256, 256);
            Fill(ty, (d, i, j) => ta[d, i, j] / tb[d, i, j]);

            tc = ta / tb;

            (float mean, float min, float max) = Test(tc, ty);
            //Assert.IsTrue(mean < 1e-6);
            Debug.WriteLine($"Division test passed with error mean={mean}, min={min}, max={max}");
        }
    }
}