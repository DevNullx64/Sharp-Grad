using SharpGrad;
using SharpGrad.Activation;
using SharpGrad.DifEngine;
using SharpGrad.DifEngine.SyntaxBuilder.CPU;
using System.Diagnostics;
using System.Numerics;

namespace TestProject.Activations
{
    [TestClass]
    public class TestReLU
    {
        public static void ReLU<T>()
            where T : struct, INumber<T>
        {
            Variable<T> a = new("a", T.CreateTruncating(1.5));
            var c = a.ReLU();
            var cpu = new DeviceCpu();
            cpu.Forward(c);
            var r = T.Max(T.Zero, T.CreateTruncating(1.5));
            Assert.AreEqual(r, c.Data[0]);

            a[0] = T.CreateTruncating(-2.0);
            cpu.Forward(c);
            r = T.Max(T.Zero, T.CreateTruncating(-2.0));
            Assert.AreEqual(r, c.Data[0]);

            for (int i = 0; i < 10; i++)
            {
                var aData = Common.Random<T>();
                a[0] = aData;
                cpu.Forward(c);
                r = T.Max(T.Zero, aData);
                Assert.AreEqual(r, c.Data[0]);
            }
        }

        [TestMethod]
        public void ReLUHalf() => ReLU<Half>();
        [TestMethod]
        public void ReLUFloat() => ReLU<float>();
        [TestMethod]
        public void ReLUDouble() => ReLU<double>();
        [TestMethod]
        public void ReLUDecimal() => ReLU<decimal>();

        [TestMethod]
        public void ReLUByte() => ReLU<byte>();
        [TestMethod]
        public void ReLUSByte() => ReLU<sbyte>();
        [TestMethod]
        public void ReLUShort() => ReLU<short>();
        [TestMethod]
        public void ReLUUShort() => ReLU<ushort>();
        [TestMethod]
        public void ReLUInt() => ReLU<int>();
        [TestMethod]
        public void ReLUUInt() => ReLU<uint>();
        [TestMethod]
        public void ReLULong() => ReLU<long>();
        [TestMethod]
        public void ReLUULong() => ReLU<ulong>();
        [TestMethod]
        public void ReLUBigInteger() => ReLU<BigInteger>();
    }
}
