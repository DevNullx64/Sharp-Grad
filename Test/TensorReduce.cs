using SharpGrad;
using SharpGrad.Tensors;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Text;
using System.Threading.Tasks;

namespace Test
{
    [TestClass]
    public class TensorReduce
    {
        [TestMethod]
        public void TestReduceSum()
        {
            Tensor<float> tensor = new TensorData<float>("Input", new Shape(3, 3), [1, 2, 3, 4, 5, 6, 7, 8, 9]);
            Tensor<float> sum = tensor.Sum();
            Assert.AreEqual(sum.Shape, new Shape(3, 1));
            Assert.AreEqual(sum[0, 0], 6);
            Assert.AreEqual(sum[1, 0], 15);
            Assert.AreEqual(sum[2, 0], 24);
        }
    }
}
