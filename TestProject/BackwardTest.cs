using SharpGrad;
using SharpGrad.DifEngine;
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

            Dimdexer dimdexer = new(dim);
            // Gradient of C3 in ((A + B) * 2) / 3 is 1

            // Gradient of C2 in ((A + B) * 2) / 3 is 2
            dimdexer.MoveNext(); Assert.AreEqual(1 / 3f, C2.Grad[dimdexer.Current]);
            dimdexer.MoveNext(); Assert.AreEqual(1 / 3f, C2.Grad[dimdexer.Current]);
            dimdexer.MoveNext(); Assert.AreEqual(1 / 3f, C2.Grad[dimdexer.Current]);

            // Gradient of C in ((A + B) * 2) / 3 is 2 / 3
            dimdexer.Reset();
            dimdexer.MoveNext(); Assert.AreEqual(2f / 3, C.Grad[dimdexer.Current]);
            dimdexer.MoveNext(); Assert.AreEqual(2f / 3, C.Grad[dimdexer.Current]);
            dimdexer.MoveNext(); Assert.AreEqual(2f / 3, C.Grad[dimdexer.Current]);

            // Gradient of A in ((A + B) * 2) / 3 is 2 / 3
            dimdexer.Reset();
            dimdexer.MoveNext(); Assert.AreEqual(2f / 3, A.Grad[dimdexer.Current]);
            dimdexer.MoveNext(); Assert.AreEqual(2f / 3, A.Grad[dimdexer.Current]);
            dimdexer.MoveNext(); Assert.AreEqual(2f / 3, A.Grad[dimdexer.Current]);

            // Gradient of B in ((A + B) * 2) / 3 is 2 / 3
            dimdexer.Reset();
            dimdexer.MoveNext(); Assert.AreEqual(2f / 3, B.Grad[dimdexer.Current]);
            dimdexer.MoveNext(); Assert.AreEqual(2f / 3, B.Grad[dimdexer.Current]);
            dimdexer.MoveNext(); Assert.AreEqual(2f / 3, B.Grad[dimdexer.Current]);
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
            Dimdexer dimdexer = new(dim);
            // Gradient of C2 in (A + B) * 2 is 1

            // Gradient of C in (A + B) * 2 is 2
            dimdexer.MoveNext(); Assert.AreEqual(2f, C.Grad[dimdexer.Current]);
            dimdexer.MoveNext(); Assert.AreEqual(2f, C.Grad[dimdexer.Current]);
            dimdexer.MoveNext(); Assert.AreEqual(2f, C.Grad[dimdexer.Current]);
            // Gradient of A in (A + B) * 2 is 2
            dimdexer.Reset();
            dimdexer.MoveNext(); Assert.AreEqual(2f, A.Grad[dimdexer.Current]);
            dimdexer.MoveNext(); Assert.AreEqual(2f, A.Grad[dimdexer.Current]);
            dimdexer.MoveNext(); Assert.AreEqual(2f, A.Grad[dimdexer.Current]);
            // Gradient of B in (A + B) * 2 is 2
            dimdexer.Reset();
            dimdexer.MoveNext(); Assert.AreEqual(2f, B.Grad[dimdexer.Current]);
            dimdexer.MoveNext(); Assert.AreEqual(2f, B.Grad[dimdexer.Current]);
            dimdexer.MoveNext(); Assert.AreEqual(2f, B.Grad[dimdexer.Current]);
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
            Dimdexer dimdexer = new(dim);
            // Gradient of C in A * 2 is 1

            // Gradient of A in A * 2 is 2
            dimdexer.MoveNext(); Assert.AreEqual(2f, A.Grad[dimdexer.Current]);
            dimdexer.MoveNext(); Assert.AreEqual(2f, A.Grad[dimdexer.Current]);
            dimdexer.MoveNext(); Assert.AreEqual(2f, A.Grad[dimdexer.Current]);
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

            Dimdexer dimdexer = new(dim);
            // Gradient of A in A + B is 1
            Assert.AreEqual(1, A.Grad[dimdexer.Current]); dimdexer.MoveNext();
            Assert.AreEqual(1, A.Grad[dimdexer.Current]); dimdexer.MoveNext();
            Assert.AreEqual(1, A.Grad[dimdexer.Current]);

            // Gradient of B in A + B is 1
            dimdexer.Reset();
            dimdexer.MoveNext(); Assert.AreEqual(1, B.Grad[dimdexer.Current]);
            dimdexer.MoveNext(); Assert.AreEqual(1, B.Grad[dimdexer.Current]);
            dimdexer.MoveNext(); Assert.AreEqual(1, B.Grad[dimdexer.Current]);

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

            Dimdexer dimdexer = new(dim);
            // Gradient of A in A - B is 1
            Assert.AreEqual(1, A.Grad[dimdexer.Current]); dimdexer.MoveNext();
            Assert.AreEqual(1, A.Grad[dimdexer.Current]); dimdexer.MoveNext();
            Assert.AreEqual(1, A.Grad[dimdexer.Current]); dimdexer.MoveNext();

            // Gradient of B in A - B is -1
            Assert.AreEqual(-1, B.Grad[dimdexer.Current]); dimdexer.MoveNext();
            Assert.AreEqual(-1, B.Grad[dimdexer.Current]); dimdexer.MoveNext();
            Assert.AreEqual(-1, B.Grad[dimdexer.Current]);

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

            Dimdexer dimdexer = new(dim);
            // Gradient of A in A * B is B
            dimdexer.MoveNext();
            Assert.AreEqual(4, A.Grad[dimdexer.Current]); dimdexer.MoveNext();
            Assert.AreEqual(5, A.Grad[dimdexer.Current]); dimdexer.MoveNext();
            Assert.AreEqual(6, A.Grad[dimdexer.Current]);

            // Gradient of B in A * B is A
            dimdexer.Reset();
            dimdexer.MoveNext();
            Assert.AreEqual(1, B.Grad[dimdexer.Current]); dimdexer.MoveNext();
            Assert.AreEqual(2, B.Grad[dimdexer.Current]); dimdexer.MoveNext();
            Assert.AreEqual(3, B.Grad[dimdexer.Current]);

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

            Dimdexer dimdexer = new(dim);
            // Gradient of A in A / B is 1 / B
            dimdexer.MoveNext();
            Assert.AreEqual(A.Grad[dimdexer.Current], 1.0f / bData[0]); dimdexer.MoveNext();
            Assert.AreEqual(A.Grad[dimdexer.Current], 1.0f / bData[1]); dimdexer.MoveNext();
            Assert.AreEqual(A.Grad[dimdexer.Current], 1.0f / bData[2]);

            // Gradient of B in A / B is -A / B^2
            dimdexer.Reset();
            dimdexer.MoveNext();
            Assert.AreEqual(B.Grad[dimdexer.Current], -aData[0] / (bData[0] * bData[0])); dimdexer.MoveNext();
            Assert.AreEqual(B.Grad[dimdexer.Current], -aData[1] / (bData[1] * bData[1])); dimdexer.MoveNext();
            Assert.AreEqual(B.Grad[dimdexer.Current], -aData[2] / (bData[2] * bData[2]));

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

            Dimdexer dimdexer = new(dim);
            // Gradient of A in A ^ B is B * A ^ (B - 1)
            dimdexer.MoveNext();
            Assert.AreEqual(4 * MathF.Pow(1, 3), A.Grad[dimdexer.Current]); dimdexer.MoveNext();
            Assert.AreEqual(5 * MathF.Pow(2, 4), A.Grad[dimdexer.Current]); dimdexer.MoveNext();
            Assert.AreEqual(6 * MathF.Pow(3, 5), A.Grad[dimdexer.Current]);

            // Gradient of B in A ^ B is A ^ B * log(A)
            dimdexer.Reset();
            dimdexer.MoveNext();
            Assert.AreEqual(MathF.Pow(1, 4) * MathF.Log(1), B.Grad[dimdexer.Current]); dimdexer.MoveNext();
            Assert.AreEqual(MathF.Pow(2, 5) * MathF.Log(2), B.Grad[dimdexer.Current]); dimdexer.MoveNext();
            Assert.AreEqual(MathF.Pow(3, 6) * MathF.Log(3), B.Grad[dimdexer.Current]);

            Console.WriteLine($"{nameof(TestPow)}({aData}, {bData}) passed: {cData}");
        }
    }
}
