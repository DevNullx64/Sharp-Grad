using SharpGrad;
using SharpGrad.DifEngine;
using SharpGrad.DifEngine.SyntaxBuilder.CPU;

namespace TestProject
{
    [TestCategory("Basic")]
    [TestClass]
    public class IndexingTest
    {
        [TestMethod]
        public void AddMatrixAndVector()
        {
            Dimension X = Common.CreateDimension(nameof(X), 3);
            Dimension Y = Common.CreateDimension(nameof(Y), 2);
            float[,] xyData = new float[3, 2] { { 1, 2 }, { 3, 4 }, { 5, 6 } };
            float[] xData = [7, 8, 9];
            float[] yData = [10, 11];
            Variable<float> xy = new(nameof(xy), xyData, X, Y);
            Variable<float> x = new(nameof(x), xData, X);
            Variable<float> y = new(nameof(y), yData, Y);

            var cpu = new DeviceCpu();

            var d = xy + x;
            cpu.Forward(d);
            var dBuffer = d.Data;
            Assert.AreEqual(dBuffer[0, 0], xyData[0, 0] + xData[0]);
            Assert.AreEqual(dBuffer[0, 1], xyData[0, 1] + xData[0]);
            Assert.AreEqual(dBuffer[1, 0], xyData[1, 0] + xData[1]);
            Assert.AreEqual(dBuffer[1, 1], xyData[1, 1] + xData[1]);
            Assert.AreEqual(dBuffer[2, 0], xyData[2, 0] + xData[2]);
            Assert.AreEqual(dBuffer[2, 1], xyData[2, 1] + xData[2]);

            var e = xy + y;
            cpu.Forward(e);
            var eBuffer = e.Data;
            Assert.AreEqual(eBuffer[0, 0], xyData[0, 0] + yData[0]);
            Assert.AreEqual(eBuffer[0, 1], xyData[0, 1] + yData[1]);
            Assert.AreEqual(eBuffer[1, 0], xyData[1, 0] + yData[0]);
            Assert.AreEqual(eBuffer[1, 1], xyData[1, 1] + yData[1]);
            Assert.AreEqual(eBuffer[2, 0], xyData[2, 0] + yData[0]);
            Assert.AreEqual(eBuffer[2, 1], xyData[2, 1] + yData[1]);

            var f = x + y;
            cpu.Forward(f);
            var fBuffer = f.Data;
            Assert.AreEqual(fBuffer[0, 0], xData[0] + yData[0]);
            Assert.AreEqual(fBuffer[0, 1], xData[0] + yData[1]);
            Assert.AreEqual(fBuffer[1, 0], xData[1] + yData[0]);
            Assert.AreEqual(fBuffer[1, 1], xData[1] + yData[1]);
            Assert.AreEqual(fBuffer[2, 0], xData[2] + yData[0]);
            Assert.AreEqual(fBuffer[2, 1], xData[2] + yData[1]);

            var g = xy + x + y;
            cpu.Forward(g);
            var gBuffer = g.Data;
            Assert.AreEqual(gBuffer[0, 0], xyData[0, 0] + xData[0] + yData[0]);
            Assert.AreEqual(gBuffer[0, 1], xyData[0, 1] + xData[0] + yData[1]);
            Assert.AreEqual(gBuffer[1, 0], xyData[1, 0] + xData[1] + yData[0]);
            Assert.AreEqual(gBuffer[1, 1], xyData[1, 1] + xData[1] + yData[1]);
            Assert.AreEqual(gBuffer[2, 0], xyData[2, 0] + xData[2] + yData[0]);
            Assert.AreEqual(gBuffer[2, 1], xyData[2, 1] + xData[2] + yData[1]);

            Console.WriteLine($"{nameof(AddMatrixAndVector)} passed.");
        }

        [TestMethod]
        public void TestIndices()
        {
            Dimension X = Common.CreateDimension(nameof(X), 3);
            Dimension Y = Common.CreateDimension(nameof(Y), 2);
            Dimension[] shape = [X, Y];

            float[,] aData = new float[,] { { 1, 2 }, { 3, 4 }, { 5, 6 } };
            Variable<float> a = new("a", aData, X, Y);
            IReadOnlyDataBuffer<float> aBuffer = (IReadOnlyDataBuffer<float>)a.Data;

            for (int x = 0; x < X.Size; x++)
            {
                for (int y = 0; y < Y.Size; y++)
                {
                    float ai = aBuffer[x, y];
                    Assert.AreEqual((x * Y.Size) + y + 1, ai);
                    Console.WriteLine($"a[{x}, {y}] = {ai}");
                }
            }
        }
    }
}
