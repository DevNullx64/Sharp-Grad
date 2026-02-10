using SharpGrad;
using SharpGrad.DifEngine;
using SharpGrad.DifEngine.SyntaxBuilder.CPU;
using System.Diagnostics;
using System.Numerics;

namespace TestProject.Operators
{
    [TestClass]
    public sealed class TestPow
    {
        public static void Pow<T>()
            where T : struct, IBinaryFloatingPointIeee754<T>
        {
            Variable<T> a = new("a", T.CreateTruncating(1.5));
            Variable<T> b = new("b", T.CreateTruncating(2.0));
            var c = a.Pow(b);
            var cpu = new DeviceCpu();
            cpu.Forward(c);
            Debug.Assert(c.Data[0] == T.Pow(T.CreateTruncating(1.5), T.CreateTruncating(2.0)));

            a[0] = T.CreateTruncating(2.0);
            b[0] = T.CreateTruncating(3.0);
            cpu.Forward(c);
            Debug.Assert(c.Data[0] == T.Pow(T.CreateTruncating(2.0), T.CreateTruncating(3.0)));

            for (int i = 0; i < 10; i++)
            {
                var aData = Common.Random<T>();
                a[0] = aData;
                var bData = Common.Random<T>();
                b[0] = bData;
                cpu.Forward(c);
                var r = T.Pow(aData, bData);
                Debug.Assert(c.Data[0] == r);
            }
        }

        [TestMethod]
        public void PowHalf() => Pow<Half>();
        [TestMethod]
        public void PowFloat() => Pow<float>();
        [TestMethod]
        public void PowDouble() => Pow<double>();
    }
}
