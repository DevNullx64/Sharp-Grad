using SharpGrad;
using SharpGrad.Activation;
using SharpGrad.DifEngine;
using SharpGrad.DifEngine.Loss;
using SharpGrad.DifEngine.SyntaxBuilder.CPU;

namespace TestProject
{
    [TestCategory("Backward")]
    [TestClass]
    public class BackwardTest
    {
        /// <summary>
        /// Test a composit function of ((A + B) * 2) / 3)
        /// Where A = [1, 2, 3] and B = [4, 5, 6].
        /// </summary>
        [TestMethod]
        public void TestCompisit3()
        {
            Dimension dim = Common.CreateDimension(nameof(dim), 3);
            var A = new Variable<float>("A", [1, 2, 3], dim);
            var B = new Variable<float>("B", [4, 5, 6], dim);
            var C = (A + B);
            var C2 = C * 2;
            var C3 = C2 / 3;

            var cpu = new DeviceCpu();
            cpu.Forward(C3);
            cpu.Backward(C3);

            // Gradient of C3 in ((A + B) * 2) / 3 is 1

            // Gradient of C2 in ((A + B) * 2) / 3 is 2
            Assert.AreEqual(1 / 3f, C2.Grad[0]);
            Assert.AreEqual(1 / 3f, C2.Grad[1]);
            Assert.AreEqual(1 / 3f, C2.Grad[2]);

            // Gradient of C in ((A + B) * 2) / 3 is 2 / 3
            Assert.AreEqual(2f / 3, C.Grad[0]);
            Assert.AreEqual(2f / 3, C.Grad[1]);
            Assert.AreEqual(2f / 3, C.Grad[2]);

            // Gradient of A in ((A + B) * 2) / 3 is 2 / 3
            Assert.AreEqual(2f / 3, A.Grad[0]);
            Assert.AreEqual(2f / 3, A.Grad[1]);
            Assert.AreEqual(2f / 3, A.Grad[2]);

            // Gradient of B in ((A + B) * 2) / 3 is 2 / 3
            Assert.AreEqual(2f / 3, B.Grad[0]);
            Assert.AreEqual(2f / 3, B.Grad[1]);
            Assert.AreEqual(2f / 3, B.Grad[2]);
        }

        /// <summary>
        /// Test a composit function of (A + B) * 2)
        /// Where A = [1, 2, 3] and B = [4, 5, 6].
        /// </summary>
        [TestMethod]
        public void TestCompisit2()
        {
            Dimension dim = Common.CreateDimension(nameof(dim), 3);
            var A = new Variable<float>("A", [1, 2, 3], dim);
            var B = new Variable<float>("B", [4, 5, 6], dim);
            var C = (A + B);
            var C2 = C * 2;
            var cpu = new DeviceCpu();
            cpu.Forward(C2);
            cpu.Backward(C2);
            // Gradient of C2 in (A + B) * 2 is 1

            // Gradient of C in (A + B) * 2 is 2
            Assert.AreEqual(2f, C.Grad[0]);
            Assert.AreEqual(2f, C.Grad[1]);
            Assert.AreEqual(2f, C.Grad[2]);
            // Gradient of A in (A + B) * 2 is 2
            Assert.AreEqual(2f, A.Grad[0]);
            Assert.AreEqual(2f, A.Grad[1]);
            Assert.AreEqual(2f, A.Grad[2]);
            // Gradient of B in (A + B) * 2 is 2
            Assert.AreEqual(2f, B.Grad[0]);
            Assert.AreEqual(2f, B.Grad[1]);
            Assert.AreEqual(2f, B.Grad[2]);
        }

        /// <summary>
        /// Test a composit function of A * 2
        /// Where A = [1, 2, 3].
        /// </summary>
        [TestMethod]
        public void TestCompisit1()
        {
            Dimension dim = Common.CreateDimension(nameof(dim), 3);
            var A = new Variable<float>("A", [1, 2, 3], dim);
            var C = A * 2;
            var cpu = new DeviceCpu();
            cpu.Forward(C);
            cpu.Backward(C);
            // Gradient of C in A * 2 is 1

            // Gradient of A in A * 2 is 2
            Assert.AreEqual(2f, A.Grad[0]);
            Assert.AreEqual(2f, A.Grad[1]);
            Assert.AreEqual(2f, A.Grad[2]);
        }

        /// <summary>
        /// Test a composit function of (A + B) * 2
        /// Where A = [1, 2, 3] and B = [4, 5, 6].
        /// </summary>
        [TestMethod]
        public void TestCompisit4()
        {
            Dimension dim = Common.CreateDimension(nameof(dim), 3);
            var A = new Variable<float>("A", [1, 2, 3], dim);
            var B = new Variable<float>("B", [4, 5, 6], dim);
            var C = (A + B) * 2;
            var cpu = new DeviceCpu();
            cpu.Forward(C);
            cpu.Backward(C);
            // Gradient of C in (A + B) * 2 is 1

            // Gradient of A in (A + B) * 2 is 2
            Assert.AreEqual(2f, A.Grad[0]);
            Assert.AreEqual(2f, A.Grad[1]);
            Assert.AreEqual(2f, A.Grad[2]);
            // Gradient of B in (A + B) * 2 is 2
            Assert.AreEqual(2f, B.Grad[0]);
            Assert.AreEqual(2f, B.Grad[1]);
            Assert.AreEqual(2f, B.Grad[2]);
        }

        /// <summary>
        /// Test a composit function of Sum(Sum(A, dim), s)
        /// Where A = [1, 2, 3], dim is the dimension of A and s is a scalar dimension.
        /// </summary>
        [TestMethod]
        public void TestSumBackward()
        {
            Dimension dim = Common.CreateDimension(nameof(dim), 3);
            Dimension s = Dimension.Scalar;
            var A = new Variable<float>("A", [1, 2, 3], dim);
            var sum = VMath.Sum(A, dim);
            var sum2 = VMath.Sum(sum, s);

            var cpu = new DeviceCpu();
            cpu.Forward(sum2);
            cpu.Backward(sum2);

            Assert.AreEqual(1f, A.Grad[0]);
            Assert.AreEqual(1f, A.Grad[1]);
            Assert.AreEqual(1f, A.Grad[2]);
            Assert.AreEqual(1f, sum.Grad[0]);
        }

        [TestMethod]
        public void TestReLUBackward()
        {
            Dimension dim = Common.CreateDimension(nameof(dim), 3);
            var A = new Variable<float>("A", [-1f, 0.5f, 2f], dim);
            var C = A.ReLU();

            var cpu = new DeviceCpu();
            cpu.Forward(C);
            cpu.Backward(C);

            Assert.AreEqual(0f, A.Grad[0]);
            Assert.AreEqual(1f, A.Grad[1]);
            Assert.AreEqual(1f, A.Grad[2]);
        }

        [TestMethod]
        public void TestMseBackward()
        {
            Dimension batch = Common.CreateDimension(nameof(batch), 3);
            var Y = new Variable<float>("Y", [1f, 2f, 3f], batch);
            var YHat = new Variable<float>("Y_hat", [2f, 4f, 6f], batch);
            var loss = Loss.MSE(Y, YHat, batch);

            var cpu = new DeviceCpu();
            cpu.Forward(loss);
            cpu.Backward(loss);

            float scale = 2f / batch.Size;
            float[] expectedY = [-1f * scale, -2f * scale, -3f * scale];
            float[] expectedYHat = [1f * scale, 2f * scale, 3f * scale];

            Assert.AreEqual(expectedY[0], Y.Grad[0], 1e-6);
            Assert.AreEqual(expectedY[1], Y.Grad[1], 1e-6);
            Assert.AreEqual(expectedY[2], Y.Grad[2], 1e-6);

            Assert.AreEqual(expectedYHat[0], YHat.Grad[0], 1e-6);
            Assert.AreEqual(expectedYHat[1], YHat.Grad[1], 1e-6);
            Assert.AreEqual(expectedYHat[2], YHat.Grad[2], 1e-6);
        }

        /// <summary>
        /// Test a composit function of (A + B) * 2
        /// Where A = [1, 2, 3] and B = [4, 5, 6].
        /// </summary>
        [TestMethod]
        public void TestMlpBackwardFiniteDifference()
        {
            Dimension batch = Common.CreateDimension(nameof(batch), 2);
            Dimension input = Common.CreateDimension(nameof(input), 2);
            Dimension output = Dimension.Scalar;

            float[,] xData = { { 0.2f, -0.1f }, { 0.4f, 0.3f } };
            float[] ygtData = { 0.3f, -0.2f };

            Variable<float> X = new("X", xData, batch, input);
            Variable<float> W = new("W", new float[] { 0.5f, -0.3f }, input);
            Variable<float> B = new("B", 0.1f);
            Variable<float> Ygt = new("Ygt", ygtData, batch);

            Value<float> mul = X * W;
            Value<float> sum = VMath.Sum(mul, input);
            Value<float> sumB = sum + B;
            Value<float> lossPerSample = Loss.MSE(sumB, Ygt, output);
            Value<float> loss = VMath.Sum(lossPerSample, batch) / batch.Size;
            loss.IsOutput = true;

            var cpu = new DeviceCpu();
            cpu.Forward(loss);
            cpu.Backward(loss);

            float eps = 1e-3f;
            var wData = (DataBuffer<float>)W.Data;
            var bData = (DataBuffer<float>)B.Data;

            float ComputeLoss()
            {
                cpu.Forward(loss);
                return loss.Data[0];
            }

            float NumericalGradWeight(int i)
            {
                float original = wData[i];

                wData[i] = original + eps;
                float lossPlus = ComputeLoss();

                wData[i] = original - eps;
                float lossMinus = ComputeLoss();

                wData[i] = original;
                return (lossPlus - lossMinus) / (2 * eps);
            }

            float NumericalGradBias()
            {
                float original = bData[0];

                bData[0] = original + eps;
                float lossPlus = ComputeLoss();

                bData[0] = original - eps;
                float lossMinus = ComputeLoss();

                bData[0] = original;
                return (lossPlus - lossMinus) / (2 * eps);
            }

            float numW0 = NumericalGradWeight(0);
            float numW1 = NumericalGradWeight(1);
            float numB0 = NumericalGradBias();

            Assert.AreEqual(numW0, W.Grad[0], 1e-3f);
            Assert.AreEqual(numW1, W.Grad[1], 1e-3f);
            Assert.AreEqual(numB0, B.Grad[0], 1e-3f);
        }

        /// <summary>
        /// Test a composit function of (A + B) * 2
        /// Where A = [1, 2, 3] and B = [4, 5, 6].
        /// </summary>
        [TestMethod]
        public void TestMlpBackwardNoNaN()
        {
            Dimension batch = Common.CreateDimension(nameof(batch), 2);
            Dimension input = Common.CreateDimension(nameof(input), 2);
            Dimension output = Dimension.Scalar;

            float[,] xData = { { 0.2f, -0.1f }, { 0.4f, 0.3f } };
            float[] ygtData = { 1f, 2f };

            Variable<float> X = new("X", xData, batch, input);
            Variable<float> W = new("W", new float[] { 0.5f, -0.3f }, input);
            Variable<float> B = new("B", 0.1f);
            Variable<float> Ygt = new("Ygt", ygtData, batch);

            Value<float> mul = X * W;
            Value<float> sum = VMath.Sum(mul, input);
            Value<float> sumB = sum + B;
            Value<float> activated = sumB.ReLU();
            Value<float> loss = Loss.MSE(activated, Ygt, output);
            loss = VMath.Sum(loss, batch) / batch.Size;
            loss.IsOutput = true;

            var cpu = new DeviceCpu();
            cpu.Forward(loss);
            cpu.Backward(loss);

            Assert.IsFalse(float.IsNaN(loss.Data[0]));
            Assert.IsFalse(float.IsInfinity(loss.Data[0]));

            var yData = (IReadOnlyDataBuffer<float>)activated.Data;
            for (int b = 0; b < batch.Size; b++)
            {
                Assert.IsFalse(float.IsNaN(yData[b]));
                Assert.IsFalse(float.IsInfinity(yData[b]));
            }
        }

        /// <summary>
        /// Test the addition operator A + B.
        /// Where A = [1, 2, 3] and B = [4, 5, 6].
        /// </summary>
        [TestMethod]
        public void TestAdd()
        {
            Dimension dim = Common.CreateDimension(nameof(dim), 3);
            float[] aData = [1, 2, 3];
            float[] bData = [4, 5, 6];
            var A = new Variable<float>("A", aData, dim);
            var B = new Variable<float>("B", bData, dim);
            var C = A + B;
            var cpu = new DeviceCpu();
            cpu.Forward(C);
            cpu.Backward(C);
            var cData = C.Data;

            // Gradient of A in A + B is 1
            Assert.AreEqual(1, A.Grad[0]);
            Assert.AreEqual(1, A.Grad[1]);
            Assert.AreEqual(1, A.Grad[2]);

            // Gradient of B in A + B is 1
            Assert.AreEqual(1, B.Grad[0]);
            Assert.AreEqual(1, B.Grad[1]);
            Assert.AreEqual(1, B.Grad[2]);

            Console.WriteLine($"{nameof(TestAdd)}({aData}, {bData}) passed: {cData}");
        }

        [TestMethod]
        public void TestSub()
        {
            Dimension dim = Common.CreateDimension(nameof(dim), 3);
            float[] aData = [1, 2, 3];
            float[] bData = [4, 5, 6];
            var A = new Variable<float>("A", aData, dim);
            var B = new Variable<float>("B", bData, dim);
            var C = A - B;
            var cpu = new DeviceCpu();
            cpu.Forward(C);
            cpu.Backward(C);
            var cData = C.Data;

            // Gradient of A in A - B is 1
            Assert.AreEqual(1, A.Grad[0]);
            Assert.AreEqual(1, A.Grad[1]);
            Assert.AreEqual(1, A.Grad[2]);

            // Gradient of B in A - B is -1
            Assert.AreEqual(-1, B.Grad[0]);
            Assert.AreEqual(-1, B.Grad[1]);
            Assert.AreEqual(-1, B.Grad[2]);

            Console.WriteLine($"{nameof(TestSub)}({aData}, {bData}) passed: {cData}");
        }

        [TestMethod]
        public void TestMul()
        {
            Dimension dim = Common.CreateDimension(nameof(dim), 3);
            float[] aData = [1, 2, 3];
            float[] bData = [4, 5, 6];
            var A = new Variable<float>("A", aData, dim);
            var B = new Variable<float>("B", bData, dim);
            var C = A * B;
            var cpu = new DeviceCpu();
            cpu.Forward(C);
            cpu.Backward(C);
            var cData = C.Data;

            // Gradient of A in A * B is B
            Assert.AreEqual(4, A.Grad[0]);
            Assert.AreEqual(5, A.Grad[1]);
            Assert.AreEqual(6, A.Grad[2]);

            // Gradient of B in A * B is A
            Assert.AreEqual(1, B.Grad[0]);
            Assert.AreEqual(2, B.Grad[1]);
            Assert.AreEqual(3, B.Grad[2]);

            Console.WriteLine($"{nameof(TestMul)}({aData}, {bData}) passed: {cData}");
        }

        [TestMethod]
        public void TestDiv()
        {
            Dimension dim = Common.CreateDimension(nameof(dim), 3);
            float[] aData = [1, 2, 3];
            float[] bData = [4, 5, 6];
            var A = new Variable<float>("A", aData, dim);
            var B = new Variable<float>("B", bData, dim);
            var C = A / B;
            var cpu = new DeviceCpu();
            cpu.Forward(C);
            cpu.Backward(C);
            var cData = C.Data;

            // Gradient of A in A / B is 1 / B
            Assert.AreEqual(1.0f / bData[0], A.Grad[0]);
            Assert.AreEqual(1.0f / bData[1], A.Grad[1]);
            Assert.AreEqual(1.0f / bData[2], A.Grad[2]);

            // Gradient of B in A / B is -A / B^2
            Assert.AreEqual(-aData[0] / (bData[0] * bData[0]), B.Grad[0]);
            Assert.AreEqual(-aData[1] / (bData[1] * bData[1]), B.Grad[1]);
            Assert.AreEqual(-aData[2] / (bData[2] * bData[2]), B.Grad[2]);

            Console.WriteLine($"{nameof(TestDiv)}({aData}, {bData}) passed: {cData}");
        }

        [TestMethod]
        public void TestPow()
        {
            Dimension dim = Common.CreateDimension(nameof(dim), 3);
            float[] aData = [1, 2, 3];
            float[] bData = [4, 5, 6];
            var A = new Variable<float>("A", aData, dim);
            var B = new Variable<float>("B", bData, dim);
            var C = A.Pow(B);
            var cpu = new DeviceCpu();
            cpu.Forward(C);
            cpu.Backward(C);
            var cData = C.Data;

            // Gradient of A in A ^ B is B * A ^ (B - 1)
            Assert.AreEqual(4 * MathF.Pow(1, 3), A.Grad[0]);
            Assert.AreEqual(5 * MathF.Pow(2, 4), A.Grad[1]);
            Assert.AreEqual(6 * MathF.Pow(3, 5), A.Grad[2]);

            // Gradient of B in A ^ B is A ^ B * log(A)
            Assert.AreEqual(MathF.Pow(1, 4) * MathF.Log(1), B.Grad[0]);
            Assert.AreEqual(MathF.Pow(2, 5) * MathF.Log(2), B.Grad[1]);
            Assert.AreEqual(MathF.Pow(3, 6) * MathF.Log(3), B.Grad[2]);

            Console.WriteLine($"{nameof(TestPow)}({aData}, {bData}) passed: {cData}");
        }

        [TestMethod]
        public void TestMlpGradientUpdateFixedWeights()
        {
            Dimension batch = Common.CreateDimension(nameof(batch), 2);
            Dimension input = Common.CreateDimension(nameof(input), 2);
            Dimension output = Dimension.Scalar;

            float[,] xData = { { 1f, -2f }, { 0.5f, 3f } };
            float[] ygtData = { 1.5f, -0.5f };

            Variable<float> X = new("X", xData, batch, input);
            Variable<float> W = new("W", new float[] { 0.25f, -0.75f }, input);
            Variable<float> B = new("B", 0.1f);
            Variable<float> Ygt = new("Ygt", ygtData, batch);

            Value<float> mul = X * W;
            Value<float> sum = VMath.Sum(mul, input);
            Value<float> sumB = sum + B;
            Value<float> loss = Loss.MSE(sumB, Ygt, output);
            loss = VMath.Sum(loss, batch) / batch.Size;
            loss.IsOutput = true;

            var cpu = new DeviceCpu();
            float lr = 1e-3f;
            var wData = (DataBuffer<float>)W.Data;
            var bData = (DataBuffer<float>)B.Data;

            for (int step = 0; step < 20; step++)
            {
                cpu.Forward(loss);
                cpu.Backward(loss);

                Assert.IsFalse(float.IsNaN(loss.Data[0]));
                Assert.IsFalse(float.IsInfinity(loss.Data[0]));

                var wGrad = (IReadOnlyDataBuffer<float>)W.Grad;
                var bGrad = (IReadOnlyDataBuffer<float>)B.Grad;

                float expectedW0 = wData[0] - lr * wGrad[0];
                float expectedW1 = wData[1] - lr * wGrad[1];
                float expectedB0 = bData[0] - lr * bGrad[0];

                wData[0] = expectedW0;
                wData[1] = expectedW1;
                bData[0] = expectedB0;

                Assert.AreEqual(expectedW0, wData[0], 1e-6f);
                Assert.AreEqual(expectedW1, wData[1], 1e-6f);
                Assert.AreEqual(expectedB0, bData[0], 1e-6f);

                cpu.ResetGradient(loss);
            }
        }
    }
}
