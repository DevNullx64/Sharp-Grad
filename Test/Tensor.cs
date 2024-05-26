using SharpGrad;
using SharpGrad.Tensors;
using System.Diagnostics;

namespace Test
{
    [TestClass]
    public class Operators
    {
        static readonly Random rnd = new();

        public static void Fill(Tensor<float> result, Func<int, int, int, float> fnc)
        {
            for (int i = 0; i < result.Shape[0]; i++)
                for (int j = 0; j < result.Shape[1]; j++)
                    for (int k = 0; k < result.Shape[2]; k++)
                        result[i, j, k] = fnc(i, j, k);
        }

        public static Tensor<float> NewRandom(params Dim[] dims)
        {
            Tensor<float> result = new(dims);
            for (int i = 0; i < dims[0]; i++)
                for (int j = 0; j < dims[1]; j++)
                    for (int k = 0; k < dims[2]; k++)
                        result[i, j, k] = (float)rnd.NextDouble() * 100;
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
        public void TestDynamic()
        {
            Tensors.Accelerator.PrintInformation(Console.Out);

            Tensor<float> ta = NewRandom(256, 256, 256);
            Tensor<float> tb = NewRandom(256, 256, 256);
            Tensor<float> tc = new(256, 256, 256);
            Tensor<float> ty = new(256, 256, 256);

            // Test dynamic operations
            Fill(ty, (i, j, k) =>
            {
                ty[i, j, k] += ta[i, j, k] - tb[i, j, k];
                ty[i, j, k] += ta[i, j, k] + tb[i, j, k];
                ty[i, j, k] += ta[i, j, k] * tb[i, j, k];
                ty[i, j, k] += ta[i, j, k] / tb[i, j, k];
                return ty[i, j, k];
            });
            Tensor<float>.DynGpu([OpCode.Sub, OpCode.Add, OpCode.Mul, OpCode.Div], ta, tb, tc);

            (float mean, float min, float max) = Test(tc, ty);
            Assert.IsTrue(mean < 1e-6 && min == 0 && max <= 1, $"mean={mean}/1e-6, min={min}/0, max={max}/1");
            Debug.WriteLine($"dynamic test passed with mean error: {mean}, min error: {min}, max error: {max}");
        }

        [TestMethod]
        public void TestAddition()
        {
            Tensors.Accelerator.PrintInformation(Console.Out);

            Tensor<float> ta = NewRandom(256, 256, 256);
            Tensor<float> tb = NewRandom(256, 256, 256);
            Tensor<float> tc = new(256, 256, 256);

            Tensor<float> ty = new(256, 256, 256);
            Fill(ty, (d, i, j) => ta[d, i, j] + tb[d, i, j]);

            tc = ta + tb;

            (float mean, float min, float max) = Test(tc, ty);
            Assert.IsTrue(mean == 0 && min == 0 && max == 0, $"mean={mean}/0, min={min}/0, max={max}/0");
            Debug.WriteLine($"Addition test passed with error mean={mean}, min={min}, max={max}");
        }

        [TestMethod]
        public void TestSubtraction()
        {
            Tensors.Accelerator.PrintInformation(Console.Out);

            Tensor<float> ta = NewRandom(256, 256, 256);
            Tensor<float> tb = NewRandom(256, 256, 256);
            Tensor<float> tc = new(256, 256, 256);

            Tensor<float> ty = new(256, 256, 256);
            Fill(ty, (d, i, j) => ta[d, i, j] - tb[d, i, j]);

            tc = ta - tb;

            (float mean, float min, float max) = Test(tc, ty);
            Assert.IsTrue(mean == 0 && min == 0 && max == 0, $"mean={mean}/0, min={min}/0, max={max}/0");
            Debug.WriteLine($"Subtraction test passed with error mean={mean}, min={min}, max={max}");
        }

        [TestMethod]
        public void TestMultiplication()
        {
            Tensors.Accelerator.PrintInformation(Console.Out);

            Tensor<float> ta = NewRandom(256, 256, 256);
            Tensor<float> tb = NewRandom(256, 256, 256);
            Tensor<float> tc = new(256, 256, 256);

            Tensor<float> ty = new(256, 256, 256);
            Fill(ty, (d, i, j) => ta[d, i, j] * tb[d, i, j]);

            tc = ta * tb;

            (float mean, float min, float max) = Test(tc, ty);
            Assert.IsTrue(mean == 0 && min == 0 && max == 0, $"mean={mean}/0, min={min}/0, max={max}/0");
            Debug.WriteLine($"Multiplication test passed with error mean={mean}, min={min}, max={max}");
        }

        [TestMethod]
        public void TestDivision()
        {
            Tensors.Accelerator.PrintInformation(Console.Out);

            Tensor<float> ta = NewRandom(256, 256, 256);
            Tensor<float> tb = NewRandom(256, 256, 256);
            Tensor<float> tc = new(256, 256, 256);

            Tensor<float> ty = new(256, 256, 256);
            Fill(ty, (d, i, j) => ta[d, i, j] / tb[d, i, j]);

            tc = ta / tb;

            (float mean, float min, float max) = Test(tc, ty);
            Assert.IsTrue(mean < 1e-6 && min == 0 && max <= 0.5, $"mean={mean}/1e-6, min={min}/0, max={max}/0.5");
            Debug.WriteLine($"Division test passed with error mean={mean}, min={min}, max={max}");
        }
    }
}