using SharpGrad;
using SharpGrad.DifEngine.Loss;
using SharpGrad.DifEngine.SyntaxBuilder.CPU;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace TestProject
{
    [TestCategory("Basic")]
    [TestClass]
    public class BaseTest
    {
        [TestMethod]
        public void TestSpan1D()
        {
            float[,] data = new float[,] { { 1, 2 }, { 3, 4 }, { 5, 6 } };
            var linearSpan = MemoryMarshal.CreateSpan(ref data[0, 0], data.Length);
            Assert.AreEqual(6, linearSpan.Length);

            float[,] data2 = new float[3, 2];
            int iflat = 0;
            for (int i = 0; i < data2.GetLength(0); i++)
            {
                for (int j = 0; j < data2.GetLength(1); j++)
                {
                    data2[i, j] = linearSpan[iflat++];
                }
            }

            for (int i = 0; i < data2.GetLength(0); i++)
            {
                for (int j = 0; j < data2.GetLength(1); j++)
                {
                    Assert.AreEqual(data2[i, j], data[i, j]);
                }
            }
        }

        [TestMethod]
        public void TestVariable()
        {
            Dimension dim = Common.CreateDimension(nameof(dim), 5);
            var data = new float[] { 1, 2, 3, 4, 5 };
            var variable = new Variable<float>("var", data, dim);
            var vData = variable.Data;
            Assert.AreEqual(5, vData.Shape.Size);
            Assert.AreEqual(1, vData[0]);
            Assert.AreEqual(2, vData[1]);
            Assert.AreEqual(3, vData[2]);
            Assert.AreEqual(4, vData[3]);
            Assert.AreEqual(5, vData[4]);
            Console.WriteLine($"{nameof(TestVariable)}({data}) passed: {vData}");
        }
        [TestMethod]
        public void TestConstant()
        {
            var data = new float[] { 1, 2, 3, 4, 5 };
            Dimension dim = Common.CreateDimension(nameof(dim), 5);
            var con = new Constant<float>(data, new Shape(dim), "con");
            var cData = con.Data;
            Assert.AreEqual(5, cData.Shape.Size);
            Assert.AreEqual(1, cData[0]);
            Assert.AreEqual(2, cData[1]);
            Assert.AreEqual(3, cData[2]);
            Assert.AreEqual(4, cData[3]);
            Assert.AreEqual(5, cData[4]);
            Console.WriteLine($"{nameof(TestConstant)}({data}) passed: {cData}");
        }
        [TestMethod]
        public void TestMSE()
        {
            Dimension batch = new(nameof(batch), 5);
            float[] yData = { 1, 2, 3, 4, 5 };
            float[] yHatData = { 1, 2, 3, 4, 5 };
            Value<float> Y = new Variable<float>("Y", yData, batch);
            Value<float> Y_hat = new Variable<float>("Y_hat", yHatData, batch);
            var cpu = new DeviceCpu();
            var loss = Loss.MSE(Y, Y_hat, batch);
            cpu.Forward(loss);
            var lossData = loss.Data;
            Assert.AreEqual(0, lossData[0]);
            Console.WriteLine($"{nameof(TestMSE)}({yData}, {yHatData}) passed: {lossData}");

            float[] yHatData2 = { 2, 3, 4, 5, 11 };
            Value<float> Y_hat2 = new Variable<float>("Y_hat2", yHatData2, batch);
            loss = Loss.MSE(Y, Y_hat2, batch);
            cpu.Forward(loss);
            lossData = loss.Data;
            Assert.AreEqual(8, lossData[0]);
            Console.WriteLine($"{nameof(TestMSE)}({yData}, {yHatData2}) passed: {lossData}");
        }
        [TestMethod]
        public void TestDimensionExtender()
        {
            Shape dim = (new("X", 2), new("Y", 3), new("Z", 4));
            Assert.AreEqual(24, dim.Size);
            Assert.IsFalse(dim.IsScalar);
            Assert.IsFalse(dim.IsVector);
            Console.WriteLine($"{nameof(TestDimensionExtender)}({dim}) passed.");
        }
    }
}
