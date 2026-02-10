using SharpGrad;
using SharpGrad.DifEngine;
using SharpGrad.DifEngine.SyntaxBuilder.CPU;
using System.Numerics;

namespace TestProject.Operators
{
    [TestClass]
    public sealed class TestSub
    {
        public static void Sub<T>()
            where T : struct, INumber<T>
        {
            Variable<T> a = new("a", T.CreateTruncating(1.0));
            Variable<T> b = new("b", T.CreateTruncating(2.0));
            var c = a - b;
            var cpu = new DeviceCpu();
            cpu.Forward(c);
            Assert.AreEqual(c.Data[0], T.CreateTruncating(1.0) - T.CreateTruncating(2.0));

            a[0] = T.CreateTruncating(5.0);
            b[0] = T.CreateTruncating(3.0);
            cpu.Forward(c);
            Assert.AreEqual(c.Data[0], T.CreateTruncating(5.0) - T.CreateTruncating(3.0));

            for (int i = 0; i < 10; i++)
            {
                var aData = Common.Random<T>();
                a[0] = aData;
                var bData = Common.Random<T>();
                b[0] = bData;
                cpu.Forward(c);
                Assert.AreEqual(c.Data[0], aData - bData);
            }
        }

        [TestMethod]
        public void SubHalf() => Sub<Half>();
        [TestMethod]
        public void SubFloat() => Sub<float>();
        [TestMethod]
        public void SubDouble() => Sub<double>();
        [TestMethod]
        public void SubDecimal() => Sub<decimal>();

        // The binary operator Add is not defined for the types 'System.Byte' and 'System.Byte'.
        //[TestMethod]
        //public void SubByte() => Sub<byte>();
        // The binary operator Add is not defined for the types 'System.Byte' and 'System.Byte'.
        //[TestMethod]
        //public void SubSByte() => Sub<sbyte>();
        [TestMethod]
        public void SubShort() => Sub<short>();
        [TestMethod]
        public void SubUShort() => Sub<ushort>();
        [TestMethod]
        public void SubInt() => Sub<int>();
        [TestMethod]
        public void SubUInt() => Sub<uint>();
        [TestMethod]
        public void SubLong() => Sub<long>();
        [TestMethod]
        public void SubULong() => Sub<ulong>();
        [TestMethod]
        public void SubBigInteger() => Sub<BigInteger>();
    }
}
