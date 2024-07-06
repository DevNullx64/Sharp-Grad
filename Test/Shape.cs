using ILGPU.Algorithms;
using SharpGrad;
using SharpGrad.Tensors;
using System.Diagnostics;
using System.Numerics;
using System.Security.Cryptography;

namespace Test
{
    [TestClass]
    public class ShapeTest
    {
        private static Random rnd = new();

        [TestMethod]
        public void TestIndex()
        {
            Shape shape = new(rnd.Next(), rnd.Next(), rnd.Next());
            int[] dims = shape[..^1];
            Assert.AreEqual(dims.Length, shape.Count - 1);
            Assert.AreEqual(dims[1], shape[1]);
            Assert.AreEqual(dims[0], shape[0]);

            dims = shape[..^2];
            Assert.AreEqual(dims.Length, shape.Count - 2);
            Assert.AreEqual(dims[0], shape[0]);

            dims = shape[..^3];
            Assert.AreEqual(dims.Length, shape.Count - 3);
        }

        [TestMethod]
        public void TestRange()
        {
            Shape shape = new(rnd.Next(), rnd.Next(), rnd.Next());
            Assert.AreEqual(shape[^1], shape[2]);
            Assert.AreEqual(shape[^2], shape[1]);
            Assert.AreEqual(shape[^3], shape[0]);
        }
    }
}