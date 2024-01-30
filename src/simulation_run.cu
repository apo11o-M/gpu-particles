i found this cuda code that handles the particle updates from another github repo project. Can you explain how this works for me? This project handles the "explosion" rather well as when I try running this project the physics is very stable.

__global__ void cudaRun(
	const SimulationScene::Particle* const particlesIn,
	SimulationScene::Particle* const particlesOut,
	SimulationScene::Collision* const collisions,
	const unsigned int numParticles,
	const float deltaTime)
{
	if (threadIdx.x == 0) 
	{ 
		d_velocityDelta = SimulationScene::pVec(0.0f);
		d_positionDelta = SimulationScene::pVec(0.0f);
		d_collisionForceDelta = 0.0f;
	}
	__syncthreads();

	const unsigned int index = blockIdx.x;
	const SimulationScene::Particle in = particlesIn[index];
	SimulationScene::Particle* const out = particlesOut + index;
	SimulationScene::Collision* const collision = collisions + index;

	const SimulationScene::pVec inVelocity = in.velocity;
	const SimulationScene::pVec inPosition = in.position + inVelocity * deltaTime;

	//get particles to check to check for collisions
	const unsigned int numOtherParticles = cudaCeil((float)numParticles / (float)blockDim.x);
	const unsigned int firstOtherParticle = threadIdx.x * numOtherParticles;
	const unsigned int lastOtherParticle = cudaMin(firstOtherParticle + numOtherParticles, numParticles);

	SimulationScene::pVec velocityDelta = SimulationScene::pVec(0.0f);
	SimulationScene::pVec positionDelta = SimulationScene::pVec(0.0f);
	float collisionDelta = 0.0f;

	//compute velocity/position based on colliding particles

	for (unsigned int i = firstOtherParticle; i < lastOtherParticle; i++)
	{
		if (i == index) continue;
		const SimulationScene::Particle other = particlesIn[i];

		//if the distance between the two circle centerpoints is less than the sum
		//of the radii, then they must be overlapping.
		const SimulationScene::pVec otherVelocity = other.velocity;
		const SimulationScene::pVec otherPos = other.position + otherVelocity * deltaTime;
		const SimulationScene::pVec delta = inPosition - otherPos;
		const float distanceSqr = cudaDotProduct(delta, delta);
		const float radiusSum = d_particleRadius + d_particleRadius;
		const float radiusSumSqr = radiusSum * radiusSum;

		//to avoid computing a square root, the squared values are compared instead
		if (distanceSqr < radiusSumSqr)
		{
			//if an overlap occurs, move the particle out of the way by half the overlapping distance
			const float distance = sqrt(distanceSqr);
			const float overlappingDistance = radiusSum - distance;
			const SimulationScene::pVec overlappingDelta = (delta / distance) * overlappingDistance;
			const SimulationScene::pVec dis = overlappingDelta * 0.5f;
			positionDelta += dis;

			//if the particles are sufficiently far enough away, update velocities too
			if (distanceSqr > 0.001f)
			{
				//elastic collision formula: https://en.wikipedia.org/wiki/Elastic_collision
				const float velocityMag = cudaDotProduct(inVelocity - otherVelocity, delta) / distanceSqr;
				velocityDelta += d_restitutionCoefficient * -velocityMag * delta;
				collisionDelta += velocityMag > 0.0f ? velocityMag : -1.0f * velocityMag;
			}
		}
	}
	#pragma unroll
	for (char i = 0; i < PVEC_DIM; i++)
	{
		atomicAdd(&d_velocityDelta[i], velocityDelta[i]);
		atomicAdd(&d_positionDelta[i], positionDelta[i]);
	}
	atomicAdd(&d_collisionForceDelta, collisionDelta);
	__syncthreads();

	//only 1 thread should do this part
	if (threadIdx.x == 0)
	{
		//velocity becomes the summed result of the potential elastic collisions
		SimulationScene::pVec outVelocity = inVelocity + d_velocityDelta;

		//dull the velocity and apply force of gravity
		outVelocity *= cudaPow(d_velocityDullingFactor, deltaTime * d_velocityDullingFactorRate);
		outVelocity += d_gravity * deltaTime;

		//position calculated based on velocity and sum of overlaps
		SimulationScene::pVec outPosition = in.position + outVelocity * deltaTime + d_positionDelta;

		//ensure the particle hasn't left the simulation bounds
		#pragma unroll
		for (char i = 0; i < PVEC_DIM; i++)
		{
			//not completely elastic; is dulled
			if (outPosition[i] < d_boundMin[i])
			{
				outPosition[i] = d_boundMin[i]; 
				outVelocity[i] *= -d_restitutionCoefficient;
				d_collisionForceDelta += sqrt(outVelocity[i] * outVelocity[i]);
			}
			else if (outPosition[i] > d_boundMax[i])
			{
				outPosition[i] = d_boundMax[i];
				outVelocity[i] *= -d_restitutionCoefficient;
				d_collisionForceDelta += sqrt(outVelocity[i] * outVelocity[i]);
			}
		}

		//finally store results
		out->position = outPosition;
		out->velocity = outVelocity;
		out->color = in.color;
		collision->force = d_collisionForceDelta;
	}
}