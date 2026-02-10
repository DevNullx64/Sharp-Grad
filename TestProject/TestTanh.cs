using SharpGrad;
using SharpGrad.Activation;
using SharpGrad.DifEngine;
using SharpGrad.DifEngine.SyntaxBuilder.CPU;
using System.Diagnostics;
using System.Numerics;

namespace TestProject.Activations
{
    [TestClass]
    public class TestTanh
    {
        public static void Tanh<T>()
            where T : struct, IBinaryFloatingPointIeee754<T>, IExponentialFunctions<T>
        {
            Variable<T> a = new("a", T.CreateTruncating(1.5));
            var c = a.Tanh();
            var cpu = new DeviceCpu();
            cpu.Forward(c);
            var r = T.Tanh(T.CreateTruncating(1.5));
            Assert.AreEqual(r, c.Data[0]);

            a[0] = T.CreateTruncating(2.0);
            cpu.Forward(c);
            r = T.Tanh(T.CreateTruncating(2.0));
            Assert.AreEqual(r, c.Data[0]);

            for (int i = 0; i < 10; i++)
            {
                var aData = Common.Random<T>();
                a[0] = aData;
                cpu.Forward(c);
                r = T.Tanh(aData);
                Assert.AreEqual(r, c.Data[0]);
            }
        }

        [TestMethod]
        public void TanhHalf() => Tanh<Half>();
        [TestMethod]
        public void TanhFloat() => Tanh<float>();
        [TestMethod]
        public void TanhDouble() => Tanh<double>();
    }

    [TestClass]
    public class TestSum
    {
        public static void Sum<T>()
            where T : struct, IBinaryFloatingPointIeee754<T>, IAdditionOperators<T, T, T>
        {
            Dimension dim1 = new("test", 3);
            Variable<T> a = new("a", [
                T.CreateTruncating(1.0),
                T.CreateTruncating(2.0),
                T.CreateTruncating(3.0)],
                dim1);

            var cpu = new DeviceCpu();

            ReducedValue<T> sum = VMath.Sum(a, dim1);
            cpu.Forward(sum);
            Debug.Assert(sum.Data[0] == T.CreateTruncating(6.0));
            Debug.WriteLine($"Test sum passed. Result: {sum.Data[0]}");

            Dimension dim2 = new("test2", 2);
            Variable<T> b = new("b", new T[,] {
                { T.CreateTruncating(1.0), T.CreateTruncating(2.0) },
                { T.CreateTruncating(3.0), T.CreateTruncating(4.0) },
                { T.CreateTruncating(5.0), T.CreateTruncating(6.0) } },
                dim1, dim2);
            ReducedValue<T> sum2 = VMath.Sum(b, dim1, dim2);
            cpu.Forward(sum2);
            Debug.Assert(sum2.Data[0] == T.CreateTruncating(21.0));
            Debug.WriteLine($"Test sum passed. Result: {sum2.Data[0]}");

            // Sum along the second dimension
            ReducedValue<T> sum3 = VMath.Sum(b, dim2);
            cpu.Forward(sum3);
            Debug.Assert(sum3.Data[0] == T.CreateTruncating(1 + 2));
            Debug.Assert(sum3.Data[1] == T.CreateTruncating(3 + 4));
            Debug.Assert(sum3.Data[2] == T.CreateTruncating(5 + 6));
            var sum3Data = sum3.Data;
            Debug.WriteLine($"Test sum along dim2 passed. Result: [{sum3Data[0]}, {sum3Data[1]}, {sum3Data[2]}]");

            ReducedValue<T> sum3bis = VMath.Sum(sum3, dim1);
            cpu.Forward(sum3bis);
            Debug.Assert(sum3bis.Data[0] == T.CreateTruncating(1 + 2 + 3 + 4 + 5 + 6));
            Debug.WriteLine($"Test sum along dim1 passed. Result: {sum3bis.Data[0]}");

            // Sum along the first dimension
            ReducedValue<T> sum4 = VMath.Sum(b, dim1);
            cpu.Forward(sum4);
            Debug.Assert(sum4.Data[0] == T.CreateTruncating(1 + 3 + 5));
            Debug.Assert(sum4.Data[1] == T.CreateTruncating(2 + 4 + 6));
            var sum4Data = sum4.Data;
            Debug.WriteLine($"Test sum along dim1 passed. Result: [{sum4Data[0]}, {sum4Data[1]}]");

            ReducedValue<T> sum4bis = VMath.Sum(sum4, dim2);
            cpu.Forward(sum4bis);
            Debug.Assert(sum4bis.Data[0] == T.CreateTruncating(1 + 3 + 5 + 2 + 4 + 6));
            Debug.WriteLine($"Test sum along dim2 passed. Result: {sum4bis.Data[0]}");
        }

        [TestMethod]
        public void SumHalf() => Sum<Half>();
        [TestMethod]
        public void SumFloat() => Sum<float>();
        [TestMethod]
        public void SumDouble() => Sum<double>();
    }
}
