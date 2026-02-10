using SharpGrad;
using SharpGrad.Activation;
using SharpGrad.DifEngine;
using SharpGrad.DifEngine.SyntaxBuilder.CPU;
using System.Diagnostics;
using System.Numerics;

namespace TestProject.Activations
{
    [TestClass]
    public class TestSigmoid
    {
        public static void Sigmoid<T>()
            where T : struct, IBinaryFloatingPointIeee754<T>, IExponentialFunctions<T>
        {
            Variable<T> a = new("a", T.CreateTruncating(1.5));
            var c = a.Sigmoid();
            var cpu = new DeviceCpu();
            cpu.Forward(c);
            var r = T.One / (T.One + T.Exp(T.CreateTruncating(-1.5)));
            Assert.AreEqual(r, c.Data[0]);

            a[0] = T.CreateTruncating(2.0);
            cpu.Forward(c);
            r = T.One / (T.One + T.Exp(T.CreateTruncating(-2.0)));
            Assert.AreEqual(r, c.Data[0]);

            for (int i = 0; i < 10; i++)
            {
                var aData = Common.Random<T>();
                a[0] = aData;
                cpu.Forward(c);
                r = T.One / (T.One + T.Exp(T.CreateTruncating(-aData)));
                Assert.AreEqual(r, c.Data[0]);
            }
        }

        [TestMethod]
        public void SigmoidHalf() => Sigmoid<Half>();
        [TestMethod]
        public void SigmoidFloat() => Sigmoid<float>();
        [TestMethod]
        public void SigmoidDouble() => Sigmoid<double>();
    }
}
