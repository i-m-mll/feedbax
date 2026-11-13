import { ModelAddressSchema, type ModelAddress } from '@/generated/studioContracts';
import { canonicalJsonV2 } from './canonicalJsonV2';

export function modelAddressText(address: ModelAddress): string {
  return canonicalJsonV2(ModelAddressSchema.parse(address));
}

export function modelAddressFromText(text: string): ModelAddress {
  const address = ModelAddressSchema.parse(JSON.parse(text));
  if (modelAddressText(address) !== text) {
    throw new Error('Model address text must be canonical JSON');
  }
  return address;
}
