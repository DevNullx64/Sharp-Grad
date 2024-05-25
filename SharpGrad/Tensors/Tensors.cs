using ILGPU;
using ILGPU.Runtime;
using System.Diagnostics;

namespace SharpGrad.Tensors
{
    public static class Tensors
    {
        private static Context GetContext()
        {
            Context result = Context.Create(builder => builder.AllAccelerators());
            Debug.WriteLine($"Context created: {result}");
            return result;
        }
        private static readonly Context context = GetContext();

        private static Device GetDevice(Context context)
        {
            Device result = context.GetPreferredDevice(preferCPU: false);
            Debug.WriteLine($"Device created: {result}");
            return result;
        }
        private static readonly Device device = GetDevice(context);
        public static readonly Accelerator Accelerator = device.CreateAccelerator(context);

    }
}
