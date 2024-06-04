using ILGPU.Algorithms;
using SharpGrad;
using SharpGrad.Memory;
using SharpGrad.Tensors;
using System.Diagnostics;
using System.Numerics;
using System.Security.Cryptography;

namespace Test
{
    [TestClass]
    public class MemoryTest
    {
        private static Random rnd = new();

        [TestMethod]
        public void TestOOM()
        {
            List<AcceleratorBuffer<double>> allocs = [];
            int i = 1;
            while (true)
            {
                AcceleratorBuffer<double> newBlock = Acc.GetAcceleratorBuffer<double>(512 * 1024 * 1024);
                newBlock.Location = BufferLocation.Accelerator;
                newBlock.Fill(i++);
                allocs.Add(newBlock);
            }
        }
    }
}