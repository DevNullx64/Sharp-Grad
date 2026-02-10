using SharpGrad;
using SharpGrad.DifEngine;
using SharpGrad.DifEngine.SyntaxBuilder.CPU;
using System.Diagnostics;
using System.Numerics;

namespace TestProject.Operators
{
    [TestClass]
    public class TestMul
    {
        public static void Mul<T>()
            where T : struct, INumber<T>
        {
            Variable<T> a = new("a", T.CreateTruncating(1.5));
            Variable<T> b = new("b", T.CreateTruncating(2.0));
            var c = a * b;
            var cpu = new DeviceCpu();
            cpu.Forward(c);
            Debug.Assert(c.Data[0] == T.CreateTruncating(1.5) * T.CreateTruncating(2.0));

            a[0] = T.CreateTruncating(2.0);
            b[0] = T.CreateTruncating(3.0);
            cpu.Forward(c);
            Debug.Assert(c.Data[0] == T.CreateTruncating(6.0));

            for (int i = 0; i < 10; i++)
            {
                var aData = Common.Random<T>();
                a[0] = aData;
                var bData = Common.Random<T>();
                b[0] = bData;
                cpu.Forward(c);
                Debug.Assert(c.Data[0] == aData * bData);
            }
        }

        [TestMethod]
        public void MulHalf() => Mul<Half>();
        [TestMethod]
        public void MulFloat() => Mul<float>();
        [TestMethod]
        public void MulDouble() => Mul<double>();
        [TestMethod]
        public void MulDecimal() => Mul<decimal>();

        // The binary operator Add is not defined for the types 'System.Byte' and 'System.Byte'.
        //[TestMethod]
        //public void MulByte() => Mul<byte>();
        // The binary operator Add is not defined for the types 'System.Byte' and 'System.Byte'.
        //[TestMethod]
        //public void MulSByte() => Mul<sbyte>();
        [TestMethod]
        public void MulShort() => Mul<short>();
        [TestMethod]
        public void MulUShort() => Mul<ushort>();
        [TestMethod]
        public void MulInt() => Mul<int>();
        [TestMethod]
        public void MulUInt() => Mul<uint>();
        [TestMethod]
        public void MulLong() => Mul<long>();
        [TestMethod]
        public void MulULong() => Mul<ulong>();
        [TestMethod]
        public void MulBigInteger() => Mul<BigInteger>();
    }
}
