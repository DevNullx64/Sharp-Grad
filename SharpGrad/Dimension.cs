using ILGPU.IR.Values;
using SharpGrad.Formula;
using SharpGrad.Operators;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;

namespace SharpGrad
{
    public class Dimension(int size, SharedReduceCode reduce = SharedReduceCode.Mean, SharedBroadcastCode broadcast = SharedBroadcastCode.Repeat)
    {
        private readonly InternalDimension DimensionInfo = new(size, reduce, broadcast);

        public int Size => DimensionInfo.Size;
        public SharedReduceCode Reduce => DimensionInfo.Reduce;
        public SharedBroadcastCode Broadcast => DimensionInfo.Broadcast;
    }

}