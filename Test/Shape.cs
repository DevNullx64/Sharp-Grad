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
            Dim[] dims = shape[..^1];
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
            Dim dim = shape[^1];
            Assert.AreEqual(dim, shape[2]);
            dim = shape[^2];
            Assert.AreEqual(dim, shape[1]);
            dim = shape[^3];
            Assert.AreEqual(dim, shape[0]);
        }
    }
}