using SharpGrad;
using SharpGrad.DifEngine;

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
            float[,] aData = new float[,] { { 1, 2, 3 }, { 4, 5, 6 } };
            float[] bData = new float[] { 7, 8, 9 };
            float[] cData = new float[] { 10, 11 };
            Variable<float> a = new(aData, X, Y, "a");
            Variable<float> b = new(bData, X, "b");
            Variable<float> c = new(cData, Y, "c");

            var d = a + b;
            d.Forward();
            d.GetData(out float[,] dData);
            Assert.AreEqual(dData[0, 0], aData[0, 0] + bData[0]);
            Assert.AreEqual(dData[0, 1], aData[0, 1] + bData[1]);
            Assert.AreEqual(dData[0, 2], aData[0, 2] + bData[2]);
            Assert.AreEqual(dData[1, 0], aData[1, 0] + bData[0]);
            Assert.AreEqual(dData[1, 1], aData[1, 1] + bData[1]);
            Assert.AreEqual(dData[1, 2], aData[1, 2] + bData[2]);

            var e = a + c;
            e.Forward();
            e.GetData(out float[,] eData);
            Assert.AreEqual(eData[0, 0], aData[0, 0] + cData[0]);
            Assert.AreEqual(eData[0, 1], aData[0, 1] + cData[0]);
            Assert.AreEqual(eData[0, 2], aData[0, 2] + cData[0]);
            Assert.AreEqual(eData[1, 0], aData[1, 0] + cData[1]);
            Assert.AreEqual(eData[1, 1], aData[1, 1] + cData[1]);
            Assert.AreEqual(eData[1, 2], aData[1, 2] + cData[1]);

            var f = b + c;
            f.Forward();
            f.GetData(out float[,] fData);
            Assert.AreEqual(fData[0, 0], bData[0] + cData[0]);
            Assert.AreEqual(fData[0, 1], bData[0] + cData[1]);
            Assert.AreEqual(fData[1, 0], bData[1] + cData[0]);
            Assert.AreEqual(fData[1, 1], bData[1] + cData[1]);
            Assert.AreEqual(fData[2, 0], bData[2] + cData[0]);
            Assert.AreEqual(fData[2, 1], bData[2] + cData[1]);

            var g = a + b + c;
            g.Forward();
            g.GetData(out float[,] gData);
            Assert.AreEqual(gData[0, 0], aData[0, 0] + bData[0] + cData[0]);
            Assert.AreEqual(gData[0, 1], aData[0, 1] + bData[0] + cData[1]);
            Assert.AreEqual(gData[0, 2], aData[0, 2] + bData[0] + cData[2]);
            Assert.AreEqual(gData[1, 0], aData[1, 0] + bData[1] + cData[0]);
            Assert.AreEqual(gData[1, 1], aData[1, 1] + bData[1] + cData[1]);
            Assert.AreEqual(gData[1, 2], aData[1, 2] + bData[1] + cData[2]);

            Console.WriteLine($"{nameof(AddMatrixAndVector)} passed.");
        }

        [TestMethod]
        public void TestIndices()
        {
            Dimension X = new(nameof(X), 3);
            Dimension Y = new(nameof(Y), 2);
            Dimension[] shape = [X, Y];

            float[,] aData = new float[,] { { 1, 2 }, { 3, 4 }, { 5, 6 } };
            Variable<float> a = new(aData, X, Y, "a");

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
