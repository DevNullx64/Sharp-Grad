using SharpGrad;
using SharpGrad.DifEngine;
using SharpGrad.DifEngine.SyntaxBuilder.CPU;

namespace TestProject
{
    [TestCategory("Basic")]
    [TestClass]
    public class DimdicesTest
    {
        [TestMethod]
        public void AddMatrixAndVector()
        {
            Dimension X = new(nameof(X), 3);
            Dimension Y = new(nameof(Y), 2);
            float[,] aData = new float[,] { { 1, 2 }, { 3, 4 }, { 5, 6 } };
            float[] bData = [7, 8, 9];
            float[] cData = [10, 11];
            Variable<float> a = new("a", aData, X, Y);
            Variable<float> b = new("b", bData, X);
            Variable<float> c = new("c", cData, Y);

            var cpu = new DeviceCpu();

            var d = a + b;
            cpu.Forward(d);
            var dBuffer = d.Data;
            Assert.AreEqual(dBuffer[0, 0], aData[0, 0] + bData[0]);
            Assert.AreEqual(dBuffer[0, 1], aData[0, 1] + bData[1]);
            Assert.AreEqual(dBuffer[0, 2], aData[0, 2] + bData[2]);
            Assert.AreEqual(dBuffer[1, 0], aData[1, 0] + bData[0]);
            Assert.AreEqual(dBuffer[1, 1], aData[1, 1] + bData[1]);
            Assert.AreEqual(dBuffer[1, 2], aData[1, 2] + bData[2]);

            var e = a + c;
            cpu.Forward(e);
            var eBuffer = e.Data;
            Assert.AreEqual(eBuffer[0, 0], aData[0, 0] + cData[0]);
            Assert.AreEqual(eBuffer[0, 1], aData[0, 1] + cData[0]);
            Assert.AreEqual(eBuffer[0, 2], aData[0, 2] + cData[0]);
            Assert.AreEqual(eBuffer[1, 0], aData[1, 0] + cData[1]);
            Assert.AreEqual(eBuffer[1, 1], aData[1, 1] + cData[1]);
            Assert.AreEqual(eBuffer[1, 2], aData[1, 2] + cData[1]);

            var f = b + c;
            cpu.Forward(f);
            var fBuffer = f.Data;
            Assert.AreEqual(fBuffer[0, 0], bData[0] + cData[0]);
            Assert.AreEqual(fBuffer[0, 1], bData[0] + cData[1]);
            Assert.AreEqual(fBuffer[1, 0], bData[1] + cData[0]);
            Assert.AreEqual(fBuffer[1, 1], bData[1] + cData[1]);
            Assert.AreEqual(fBuffer[2, 0], bData[2] + cData[0]);
            Assert.AreEqual(fBuffer[2, 1], bData[2] + cData[1]);

            var g = a + b + c;
            cpu.Forward(g);
            var gBuffer = g.Data;
            Assert.AreEqual(gBuffer[0, 0], aData[0, 0] + bData[0] + cData[0]);
            Assert.AreEqual(gBuffer[0, 1], aData[0, 1] + bData[0] + cData[1]);
            Assert.AreEqual(gBuffer[0, 2], aData[0, 2] + bData[0] + cData[2]);
            Assert.AreEqual(gBuffer[1, 0], aData[1, 0] + bData[1] + cData[0]);
            Assert.AreEqual(gBuffer[1, 1], aData[1, 1] + bData[1] + cData[1]);
            Assert.AreEqual(gBuffer[1, 2], aData[1, 2] + bData[1] + cData[2]);

            Console.WriteLine($"{nameof(AddMatrixAndVector)} passed.");
        }

        [TestMethod]
        public void TestIndices()
        {
            Dimension X = new(nameof(X), 3);
            Dimension Y = new(nameof(Y), 2);
            Dimension[] shape = [X, Y];

            float[,] aData = new float[,] { { 1, 2 }, { 3, 4 }, { 5, 6 } };
            Variable<float> a = new("a", aData, X, Y);

            Dimdexer dimdexer = new(shape);
            foreach (Dimdices i in dimdexer)
            {
                int x = i[X];
                int y = i[Y];

                float ai = a[i];
                Assert.AreEqual((x * Y.Size) + y + 1, ai);
                Console.WriteLine($"a{i} = {ai}");
            }
        }
    }
}
