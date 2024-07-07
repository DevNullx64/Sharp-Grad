// #define CPU_ACCELERATOR
using ILGPU;
using ILGPU.Runtime;
using SharpGrad.Memory;
using SharpGrad.Tensors.Operators;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;

namespace SharpGrad.Tensors
{
    public partial class KernelProcessUnit : IDisposable
    {
        public static KernelProcessUnit DefaultKPU = new();

        protected readonly Context context;
        protected readonly Device device;
        protected readonly Accelerator Accelerator;
        public readonly MemoryManagementUnit MMU;
        private bool disposedValue;

        public KernelProcessUnit()
        {
            context = Context.Create(builder => builder.AllAccelerators());
#if CPU_ACCELERATOR
            device = context.GetPreferredDevice(preferCPU: true);
            Accelerator = device.CreateAccelerator(context);
#else
            device = context.GetPreferredDevice(preferCPU: false);
            Accelerator = device.CreateAccelerator(context);
#endif
            MMU = new MemoryManagementUnit(Accelerator);
        }

        public void Synchronize() => Accelerator.Synchronize();
        public void PrintInformation(TextWriter writer) { Accelerator.PrintInformation(writer); }

        #region Dispose
        protected virtual void Dispose(bool disposing)
        {
            if (!disposedValue)
            {
                if (disposing)
                {
                    Accelerator.Dispose();
                    context.Dispose();
                }

                disposedValue = true;
            }
        }

        // // TODO: substituer le finaliseur uniquement si 'Dispose(bool disposing)' a du code pour libérer les ressources non managées
        // ~KernelProcessUnit()
        // {
        //     // Ne changez pas ce code. Placez le code de nettoyage dans la méthode 'Dispose(bool disposing)'
        //     Dispose(disposing: false);
        // }

        public void Dispose()
        {
            // Ne changez pas ce code. Placez le code de nettoyage dans la méthode 'Dispose(bool disposing)'
            Dispose(disposing: true);
            GC.SuppressFinalize(this);
        }
        #endregion
    }
}