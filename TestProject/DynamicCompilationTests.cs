using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpGrad.DifEngine.CPU;

namespace TestProject.Compilation
{
    [TestClass]
    public class DynamicCompilationTests
    {
        [TestMethod]
        public void DLLCompiler_CanBeConstructed()
        {
            var compiler = new DLLCompiler();
            Assert.IsNotNull(compiler);
        }
    }
}
