using SharpGrad;
using SharpGrad.DifEngine;
using SharpGrad.NN;

namespace TestProject
{
    [TestCategory("Basic")]
    [TestClass]
    public class BaseTest
    {
        [TestMethod]
        public void TestVariable()
        {
            Dimension dim = new(nameof(dim), 5);
            var data = new float[] { 1, 2, 3, 4, 5 };
            var var = new Variable<float>(data, dim, "var");
            var.GetData(out float[] vData);
            Assert.AreEqual(5, vData.Length);
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
            Dimension dim = new(nameof(dim), 5);
            var con = new Constant<float>(data, dim, "con");
            con.GetData(out float[] cData);
            Assert.AreEqual(5, cData.Length);
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
            Value<float> Y = new Variable<float>(yData, batch, "Y");
            Value<float> Y_hat = new Variable<float>(yHatData, batch, "Y_hat");
            var loss = Loss.MSE(Y, Y_hat, batch);
            loss.Forward();
            loss.GetData(out float[] lossData);
            Assert.AreEqual(0, lossData[0]);
            Console.WriteLine($"{nameof(TestMSE)}({yData}, {yHatData}) passed: {lossData}");

            loss = Loss.MSE(Y, Y_hat, batch);
            loss.Forward();
            loss.GetData(out lossData);
            Assert.AreEqual(8, lossData[0]);
            Console.WriteLine($"{nameof(TestMSE)}({yData}, {yHatData}) passed: {lossData}");
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
