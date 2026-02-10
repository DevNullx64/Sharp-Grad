using SharpGrad;
using SharpGrad.Activation;
using SharpGrad.DifEngine;
using SharpGrad.DifEngine.SyntaxBuilder.CPU;
using System.Diagnostics;
using System.Numerics;

namespace TestProject.Activations
{
    [TestClass]
    public class TestLeakyReLU
    {
        public static void LeakyReLU<T>()
            where T : struct, INumber<T>
        {
            Variable<T> a = new("a", T.CreateTruncating(1.5));
            var c = a.LeakyReLU(T.CreateTruncating(0.1));
            var cpu = new DeviceCpu();
            cpu.Forward(c);
            var r = T.Max(T.CreateTruncating(0.1) * T.CreateTruncating(1.5), T.CreateTruncating(1.5));
            Assert.AreEqual(r, c.Data[0]);
            a[0] = T.CreateTruncating(-2.0);
            cpu.Forward(c);
            r = T.Max(T.CreateTruncating(0.1) * T.CreateTruncating(-2.0), T.CreateTruncating(-2.0));
            Assert.AreEqual(r, c.Data[0]);
            for (int i = 0; i < 10; i++)
            {
                var aData = Common.Random<T>();
                a[0] = aData;
                cpu.Forward(c);
                r = T.Max(T.CreateTruncating(0.1) * aData, aData);
                Assert.AreEqual(r, c.Data[0]);
            }
        }

        [TestMethod]
        public void LeakyReLUHalf() => LeakyReLU<Half>();
        [TestMethod]
        public void LeakyReLUFloat() => LeakyReLU<float>();
        [TestMethod]
        public void LeakyReLUDouble() => LeakyReLU<double>();

    }
}
